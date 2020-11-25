'''
Some info on various layers, so you know what to expect
depending on which layer you choose:

layer 1: wavy
layer 2: lines
layer 3: boxes
layer 4: circles?
layer 6: dogs, bears, cute animals.
layer 7: faces, buildings
layer 8: fish begin to appear, frogs/reptilian eyes.
layer 10: Monkies, lizards, snakes, duck

Choosing various parameters like num_iterations, rescale,
and num_repeats really varies on which layer you're doing.


We could probably come up with some sort of formula. The
deeper the layer is, the more iterations and
repeats you will want.

Layer 3: 20 iterations, 0.5 rescale, and 8 repeats is decent start
Layer 10: 40 iterations and 25 repeats is good. 
'''

from deepdreamer import model, load_image, tofloat32, recursive_optimize, resize_image
import numpy as np
import PIL.Image
import cv2


cap = cv2.VideoCapture('video/crime.mp4')
if cap.isOpened() == False:
    print("Error opening video stream or file")

dream_name = 'crime_out_cool'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
count = 0
skipToFrame = 1


def dream(frame, layer):
    img_result = frame
    layer_tensor = model.layer_tensors[layer]
    img_result = recursive_optimize(layer_tensor=layer_tensor, image=img_result,
                                    # how clear is the dream vs original image
                                    num_iterations=20, step_size=1.0, rescale_factor=0.5,
                                    # How many "passes" over the data. More passes, the more granular the gradients will be.
                                    num_repeats=8, blend=0.2, tile_size=50)
    img_result = np.clip(img_result, 0.0, 255.0)
    img_result = img_result.astype(np.uint8)
    return img_result


try:
    while cap.isOpened():

        count = count + 1

        ret, frame = cap.read()
        if not ret:
            cap.release()
            print("Video done! Hurray!")
            exit(0)
        if count < skipToFrame:
            continue
        print("Processing frame %d" % count)

        if cv2.countNonZero(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) == 0:
            print("Image is black: frame %d" % count)
            continue

        frame = dream(frame, 7)
        result = PIL.Image.fromarray(frame, mode='RGB')

        result.save('out/{} - frame - {}.png'.format(dream_name, count))
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    print("Video interuppted! Hope it still works :|")
    exit(0)

# When everything done, release the video capture object
cap.release()
