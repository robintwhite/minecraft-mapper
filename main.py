import cv2
import pytesseract
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
from google.cloud import vision
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

screenshot_dir = 'Screenshots'
text_dir = 'TextFiles'

if not os.path.exists('Processed'):
    os.mkdirs('Processed')
processed_dir = 'Processed'
if not os.path.exists('output'):
    os.mkdirs('output')
output_dir = 'output'

ap = argparse.ArgumentParser()
ap.add_argument('-ak', '--api_key', help='Location of Google Vision API key', required=True)
args = ap.parse_args()
vision_api_key_location = args.api_key


def get_string(img, config=r'--psm 3'):
    string = pytesseract.image_to_string(img, lang='eng', config=config)
    return string


def clean_string(string):
    tmp = ''
    for line in string:
        tmp += line.replace('\n', ' ')  # strip('\n') #
    tmp = tmp.replace("'", '"')
    return tmp


def image_resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


def process_image(img):
    thresh_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(thresh_img, 220, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.GaussianBlur(thresh_img, (5, 5), 0)
    thresh_img = cv2.bilateralFilter(thresh_img, 5, 50, 10)
    _, thresh_img = cv2.threshold(thresh_img, 50, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
    return thresh_img


def get_coords(data):
    p = re.compile(r'-?\d+\.\d+\s?\/\s?-?\d+\.\d+\s?\/\s?-?\d+\.\d+')
    coords = p.findall(data)
    if coords:
        coords = re.findall(r'-?\d+\.\d+', coords[0])
        coords = [float(x) for x in coords]
    else:
        pass
    return coords


def get_biome(data):
    p = re.compile(r'Biome\: minecraft\:?(\w+)')
    biome = p.findall(data)
    return biome


def get_string_gvision(processed_img, api_key=vision_api_key_location):
    client = vision.ImageAnnotatorClient.from_service_account_file(api_key)
    success, encoded_image = cv2.imencode('.png', processed_img)
    image = vision.types.Image(content=encoded_image.tobytes())
    text_response = client.text_detection(image=image)
    string_Gvision = text_response.text_annotations[0].description
    string_Gvision = clean_string(string_Gvision)
    # print(string_Gvision)
    return string_Gvision


def do_steps(img, entry):
    coords = None
    biomes = None
    processed_img = process_image(img)
    string_vision = get_string_gvision(processed_img)
    coord = get_coords(string_vision)
    biome = get_biome(string_vision)
    if coord:
        coords = coord
        if biome:
            biomes = biome[0]
        else:
            biomes = 'unknown'
    return entry.name, string_vision, coords, biomes


def create_dataframe():
    df = None
    try:  # check if already a database json file
        df = pd.read_json(os.path.join('output', 'dataframe.json'), orient='records')
    except ValueError:
        print('No dataframe found, creating new.')
    if df is None:
        print('[INFO] Creating dataframe...')
        df = pd.DataFrame(columns=['Screenshot', 'Text', 'Coords_x', 'Coords_y', 'Coords_z', 'Biomes'])
        for entry in os.scandir(screenshot_dir):
            if entry.path.endswith(".png") and entry.is_file():
                img = cv2.imread(entry.path)
                entry_name, text, coords, biomes = do_steps(img, entry)
                img_thumb = image_resize(img, 270)
                if coords:
                    print('[INFO] Adding new entry: {}'.format(entry.name))
                    df = df.append(
                        {'Screenshot': [entry_name], 'Text': text, 'Coords_x': coords[0], 'Coords_y': coords[1],
                         'Coords_z': coords[2], 'Biomes': biomes,
                         'Image': cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB).astype(np.uint8)}, ignore_index=True)
                else:
                    print('[INFO] no coordinates found: {}'.format(entry.name))

    else:  # df already exists, check what images don't have entries and append
        for entry in os.scandir(screenshot_dir):
            if entry.path.endswith(".png") and entry.is_file():
                if entry.name in df.Screenshot.str[0].values:
                    print('[INFO] Entry already exists: {}'.format(entry.name))
                else:
                    print('[INFO] Adding new entry: {}'.format(entry.name))
                    img = cv2.imread(entry.path)
                    entry_name, text, coords, biomes = do_steps(img, entry)
                    img_thumb = image_resize(img, 270)
                    if coords:
                        df = df.append(
                            {'Screenshot': [entry_name], 'Text': text, 'Coords_x': coords[0], 'Coords_y': coords[1],
                             'Coords_z': coords[2], 'Biomes': biomes,
                             'Image': cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB)}, ignore_index=True)
                    else:
                        print('[INFO] no coordinates found: {}'.format(entry.name))

    df.to_json(os.path.join('output', 'dataframe.json'), orient='records')
    return df


def hover(event):
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w, h = fig.get_size_inches() * fig.dpi
        ws = (event.x > w / 2.) * -1 + (event.x <= w / 2.)
        hs = (event.y > h / 2.) * -1 + (event.y <= h / 2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0] * ws, xybox[1] * hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy = (x[ind], y[ind])
        # set the image corresponding to that point
        im.set_data(arr[ind])
    else:
        # if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()


if __name__ == '__main__':

    df = None
    try:  # check if already a database json file
        df = pd.read_json(os.path.join('output', 'dataframe.json'), orient='records')
    except ValueError:
        print('No dataframe found, creating new.')
    if df is None:
        print('[INFO] Creating dataframe...')
        df = pd.DataFrame(columns=['Screenshot', 'Text', 'Coords_x', 'Coords_y', 'Coords_z', 'Biomes'])
        for entry in os.scandir(screenshot_dir):
            if entry.path.endswith(".png") and entry.is_file():
                img = cv2.imread(entry.path)
                entry_name, text, coords, biomes = do_steps(img, entry)
                img_thumb = image_resize(img, 270)
                if coords:
                    print('[INFO] Adding new entry: {}'.format(entry.name))
                    df = df.append(
                        {'Screenshot': [entry_name], 'Text': text, 'Coords_x': coords[0], 'Coords_y': coords[1],
                         'Coords_z': coords[2], 'Biomes': biomes,
                         'Image': cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB).astype(np.uint8)}, ignore_index=True)
                else:
                    print('[INFO] no coordinates found: {}'.format(entry.name))

    else:  # df already exists, check what images don't have entries and append
        for entry in os.scandir(screenshot_dir):
            if entry.path.endswith(".png") and entry.is_file():
                if entry.name in df.Screenshot.str[0].values:
                    print('[INFO] Entry already exists: {}'.format(entry.name))
                else:
                    print('[INFO] Adding new entry: {}'.format(entry.name))
                    img = cv2.imread(entry.path)
                    entry_name, text, coords, biomes = do_steps(img, entry)
                    img_thumb = image_resize(img, 270)
                    if coords:
                        df = df.append(
                            {'Screenshot': [entry_name], 'Text': text, 'Coords_x': coords[0], 'Coords_y': coords[1],
                             'Coords_z': coords[2], 'Biomes': biomes,
                             'Image': cv2.cvtColor(img_thumb, cv2.COLOR_BGR2RGB)}, ignore_index=True)
                    else:
                        print('[INFO] no coordinates found: {}'.format(entry.name))

    df.to_json(os.path.join('output', 'dataframe.json'), orient='records')

    # Generate data x, y for scatter and an array of images.
    x = df.Coords_x
    y = df.Coords_z
    arr = df.Image.values

    # create figure and plot scatter
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111)
    line, = ax.plot(x, y, ls="", marker="o", alpha=0)

    # create the annotations box
    im = OffsetImage(arr[0], zoom=0.75)
    xybox = (50., 50.)
    ab = AnnotationBbox(im, (0, 0), xybox=xybox, xycoords='data', boxcoords="offset points", pad=0.3,
                        arrowprops=dict(arrowstyle="->"))
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)
    sns.scatterplot(x='Coords_x', y='Coords_z', hue='Biomes', size='Coords_y', data=df, ax=ax)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0., prop={'size': 8})
    plt.tight_layout()
    plt.show()
