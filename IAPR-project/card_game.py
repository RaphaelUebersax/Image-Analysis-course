import cv2
import sympy as sym
import math
import skimage
import skimage.morphology
import skimage.io
import os
import pandas as pd
from data.classifier import ValueClassifier
from data.utils import *

suite_dict = {
    0: "C",  # Club
    1: "D",  # Diamond
    2: "H",  # Heart
    3: "S"  # Spade
}

value_dict = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "J",  # Jack
    11: "Q",  # Queen
    12: "K"  # King
}


def clean_img(im):
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # blur the image for thresholding
    blur_im = cv2.GaussianBlur(gray_im, (3, 3), 0)
    # adaptive threshold OTSU
    ret, thresh_im = cv2.threshold(blur_im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # foreground is darker than background, so an inversion here
    thresh_im = 255 - thresh_im
    return thresh_im


def find_green_boxes(image, debug=False):
    image = cv2.resize(image, None, fx=rescale_factor, fy=rescale_factor, interpolation=cv2.INTER_LINEAR)  # Rescale
    hsv_fg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV
    green_mask = cv2.inRange(hsv_fg, (25, 35, 35), (95, 255, 255))  # Green mask
    blue_mask = cv2.inRange(hsv_fg, (80, 110, 110), (150, 255, 255))  # Blue mask

    ## Slice the green and the blue
    binary = green_mask > 0
    binary_2 = blue_mask > 0
    binary = binary | binary_2

    # Detect dealer chip
    dealer = skimage.morphology.opening(binary, disk1)
    dealer = skimage.morphology.closing(dealer, disk1)

    # Detect cards
    cards = binary ^ dealer
    cards = skimage.morphology.closing(cards, disk2)
    cards = skimage.img_as_ubyte(cards)

    if (debug):
        green = np.zeros_like(image)
        green[binary] = image[binary]
        return cards, dealer, green
    else:
        return cards, dealer


def crop_suite_value_MNIST(card_img):
    # Crop the suite
    suite_img = card_img[37:197, 20:135]

    # Crop the value
    value_img = card_img[171:541, 43:413]

    # Make it look like MNIST samples
    value_img = cv2.cvtColor(value_img, cv2.COLOR_BGR2GRAY)
    value_img = 255 - value_img
    _, value_img = cv2.threshold(value_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    value_img = cv2.GaussianBlur(value_img, (25, 25), 0)
    value_img = cv2.resize(value_img, (28, 28), interpolation=cv2.INTER_CUBIC)
    value_img = cv2.blur(value_img, (2, 2), 0)

    return suite_img, value_img


def segmentation(cards_im, dealer_im, original_im):
    # get height and width of image
    height, width = original_im.shape[0], original_im.shape[1]

    # We use RETR_EXTERNAL mode to find external contour.
    contour = cv2.findContours(cards_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Keep the biggest contour only
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    contours_list = []
    for i in range(4):
        contours_list.append(contour[i][:, 0, :])

    contours_card = []
    M_card = []
    upper_left = []
    upper_right = []
    crop_imgs = []
    angles = []
    sizes = []
    for contour in contours_list:
        # Compute convex hull of contour
        hull = cv2.convexHull(contour)[:, 0, :]
        hull = (1 / rescale_factor * hull).astype(int)

        # Make sure contour is continuous
        if np.any(hull[0, :] != hull[-1, :]):
            hull = np.vstack((hull, hull[0, :]))

        ##crop the tilted card as rectangle img
        # get min rect area enclosing contour
        rect = cv2.minAreaRect((1 / rescale_factor * contour).astype(int))
        center, size, angle = rect[0], rect[1], rect[2]
        angles.append(angle)
        sizes.append(size)
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # return tuple type: (center(x, y), (width, height), angle of rotation) = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        # keep box points for plotting function
        box = np.int0(box)
        # calculate the rotation matrix of card
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image by the calculated matrix
        original_im_rot = cv2.warpAffine(original_im, M, (width, height))
        # crop the rotated rectangle is vertical to the card in for loop
        crop_img = cv2.getRectSubPix(original_im_rot, size, center)

        if crop_img.shape[0] < crop_img.shape[1]:
            crop_img_rescaled = cv2.resize(crop_img, (712, 466), interpolation=cv2.INTER_CUBIC)
        else:
            crop_img_rescaled = cv2.resize(crop_img, (466, 712), interpolation=cv2.INTER_CUBIC)
        crop_imgs.append(crop_img_rescaled)

        upper_left_corner = np.argmin(np.sum(box, axis=1))
        diag_vect_left = np.array([rect[0][0] - box[upper_left_corner, 0], rect[0][1] - box[upper_left_corner, 1]])

        if upper_left_corner == 3:
            upper_right_corner = 0
        else:
            upper_right_corner = upper_left_corner + 1

        # Diagonal vector to crop the suit
        diag_vect_right = np.array([rect[0][0] - box[upper_right_corner, 0], rect[0][1] - box[upper_right_corner, 1]])

        # Keep only the four corners of the bounding box
        box = np.vstack((box, box[0, :]))
        center = rect[0]
        M_card.append(center)
        contours_card.append(box)
        upper_left.append(diag_vect_left)
        upper_right.append(diag_vect_right)

    # If contour is not a rectange (circle in our case)
    dealer_im = skimage.img_as_ubyte(dealer_im)
    # We use RETR_EXTERNAL mode to find external contour.
    contour_deal = cv2.findContours(dealer_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Keep the biggest contour only
    contour_deal = contour_deal[0][:, 0, :]
    hull = cv2.convexHull(contour_deal)[:, 0, :]
    hull = (1 / rescale_factor * hull).astype(int)
    contour_dealer = hull
    M = cv2.moments(hull)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    M_dealer = np.array([cX, cY])

    # Assign moments to respective players
    M_card_array = np.array(M_card)

    # sort player number
    player_one = np.argmax(M_card_array[:, 1])  # max y in upper left corresponds to player 1 card upper left corner
    player_two = np.argmax(M_card_array[:, 0])  # max x in upper right corresponds to player 2 card upper left corner
    player_three = np.argmin(M_card_array[:, 1])  # min y in upper left corresponds to player 3 card upper left corner
    player_four = np.argmin(M_card_array[:, 0])  # min x in upper right corresponds to player 4 card upper left corner

    # Segment player 1
    M_player_one = M_card_array[player_one, :]
    diag_one = upper_left[player_one]

    if sizes[player_one][0] < sizes[player_one][1]:
        crop_one = crop_imgs[player_one]
    elif sizes[player_one][0] >= sizes[player_one][1] and angles[player_one] < 0:
        crop_one = cv2.rotate(crop_imgs[player_one], cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        crop_one = cv2.rotate(crop_imgs[player_one], cv2.ROTATE_90_CLOCKWISE)

    suit_one, crop_value_one = crop_suite_value_MNIST(crop_one)

    # Segment player 2
    M_player_two = M_card_array[player_two, :]
    diag_two = upper_right[player_two]

    if sizes[player_two][0] > sizes[player_two][1]:
        crop_two = cv2.rotate(crop_imgs[player_two], cv2.ROTATE_90_CLOCKWISE)
    elif sizes[player_two][0] <= sizes[player_two][1] and angles[player_two] < 0:
        crop_two = crop_imgs[player_two]
    else:
        crop_two = cv2.rotate(crop_imgs[player_two], cv2.ROTATE_180)

    suit_two, crop_value_two = crop_suite_value_MNIST(crop_two)

    # Segment player 3
    M_player_three = M_card_array[player_three, :]
    diag_three = upper_left[player_three]

    if sizes[player_three][0] < sizes[player_three][1]:
        crop_three = cv2.rotate(crop_imgs[player_three], cv2.ROTATE_180)
    elif sizes[player_three][0] >= sizes[player_three][1] and angles[player_three] < 0:
        crop_three = cv2.rotate(crop_imgs[player_three], cv2.ROTATE_90_CLOCKWISE)
    else:
        crop_three = cv2.rotate(crop_imgs[player_three], cv2.ROTATE_90_COUNTERCLOCKWISE)

    suit_three, crop_value_three = crop_suite_value_MNIST(crop_three)

    # Segment player 4
    M_player_four = M_card_array[player_four, :]
    diag_four = upper_right[player_four]

    if sizes[player_four][0] > sizes[player_four][1]:
        crop_four = cv2.rotate(crop_imgs[player_four], cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif sizes[player_four][0] <= sizes[player_four][1] and angles[player_four] < 0:
        crop_four = cv2.rotate(crop_imgs[player_four], cv2.ROTATE_180)
    else:
        crop_four = crop_imgs[player_four]

    suit_four, crop_value_four = crop_suite_value_MNIST(crop_four)

    # Combine the result
    M_player = [M_player_one, M_player_two, M_player_three, M_player_four]
    diag_card = [diag_one, diag_two, diag_three, diag_four]
    suit_img = [suit_one, suit_two, suit_three, suit_four]
    value_img = [crop_value_one, crop_value_two, crop_value_three, crop_value_four]

    M_suit = []

    for diag, M in zip(diag_card, M_player):
        M_suit_y = M[0] - (int)(diag[0] * 6.8 / 10)
        M_suit_x = M[1] - (int)(diag[1] * 6.8 / 10)
        M_suit.append([M_suit_y, M_suit_x])

    # Assign the dealer based on distance to centroid representing the players
    dist = []
    for centroid in M_player:
        dist.append(np.sum(np.power((centroid - M_dealer), 2)))

    dealer_assign = [0] * 4
    dealer_assign[np.argmin(np.array(dist))] = 1

    return contour_dealer, contours_card, M_player, M_dealer, M_suit, suit_img, value_img, dealer_assign


def find_player_with_highest_card(card_value):
    winner = np.argwhere(card_value == np.amax(card_value))
    winner = winner.flatten()
    return winner


def decision(game, save_to_csv=False):
    ranks = [["0D"] * 4] * 13
    ranks = np.array(ranks)
    dealer = []
    standard_score = np.zeros(4)
    advanced_score = np.zeros(4)

    # Parse game into rank matrix
    for g_idx in range(len(game)):
        card_matrix = game[g_idx]

        for p_idx in range(card_matrix.shape[1]):
            ranks[g_idx, p_idx] = "{}{}".format(value_dict[card_matrix[0, p_idx]],
                                                suite_dict[card_matrix[1, p_idx]])

    for card_matrix in game:
        # Standard Game
        winner = find_player_with_highest_card(card_matrix[0, :])
        standard_score[winner] += 1

        # Advanced Game
        # Find dealer and his suite
        dealer_idx = np.argmax(card_matrix[2, :])
        active_suite = card_matrix[1, dealer_idx]
        dealer.append(dealer_idx + 1)  # idx 0 is player 1 and so on

        # Copy game_card_value
        game_card_value = card_matrix[0, :].copy()

        # Filter GGed players
        GG_players = np.where(card_matrix[1, :] != active_suite)
        game_card_value[GG_players] = -1

        # Find winner
        winner = find_player_with_highest_card(game_card_value)
        advanced_score[winner] += 1

    # Save to csv
    if save_to_csv:
        ranks_df = pd.DataFrame(ranks, columns=['P1', 'P2', 'P3', 'P4'])
        dealer_df = pd.DataFrame(dealer, columns=['D'])
        game_df = ranks_df.join(dealer_df)
        game_df.index += 1
        game_df.to_csv(r'./game.csv', header=True, index=True)

    return ranks, list(dealer), list(standard_score.astype(int)), list(advanced_score.astype(int))


def discriminant_function(x, mu, cov, prob_class):
    """
    Function that compute the discriminatn function g(x) of one class
    """
    inv_cov = np.linalg.inv(cov)
    g = -1.0 * np.matmul(np.matmul(x - mu, inv_cov), (x - mu)) + np.log(prob_class) \
        + np.log(1.0 / ((2 * math.pi) ** (mu.shape[0] / 2.0) * np.linalg.det(cov) ** (1 / 2)))
    return g


def Bayes_Classification(all_data):
    """
    Function that computes the analytical expression of the separation curve for the bayes classification under assumption
    of normal distributions.

    Parameters
    ----------
    all_data : list of numpy.ndarray
        All the data as a list where each element correspond to a 2d-array of the data of one class.

    Returns
    -------
    decision_boundary : list of sympy.core.add.Add
        Symbolique expression for x1 and x2, that expresses the boundary between two classes
    g: list of sympy.core.add.Add
        Symbolique expression for x1 and x2, that expresses the discriminant function of one class
    """
    # Compute mean, covariance and number of data of each class
    mu, cov, prob_class = [], [], []
    nb_classes = len(all_data)
    for data in all_data:
        mean = np.mean(data, axis=0)
        mu.append(mean)
        cov.append(1 / (data.shape[0] - 1) * np.matmul((data - mean).transpose(), (data - mean)))
        prob_class.append(data.shape[0])

    # Convert the nb of data of each class into its probability by diving by the total amout of data
    prob_class = np.array(prob_class)
    prob_class = prob_class / np.sum(prob_class)

    # Compute discriminant function
    g = []
    x1 = sym.Symbol('x1', real=True)
    x2 = sym.Symbol('x2', real=True)
    for i in range(nb_classes):
        g_i = discriminant_function([x1, x2], mu[i], cov[i], prob_class[i])
        g_i = sym.expand(g_i)
        g.append(g_i)

    # Compute the decision hyperplane
    decision_boundary = []
    for i in range(nb_classes - 1):
        for j in range(i + 1, nb_classes):
            decision_boundary.append(g[i] - g[j])

    return decision_boundary, g


def get_fourier_descriptors(im):
    # preprocessing to remove blue&green contours
    hsv_fg = skimage.color.rgb2hsv(im)
    hsv_fg = skimage.img_as_ubyte(hsv_fg)
    green_mask = cv2.inRange(hsv_fg, (45, 30, 30), (115, 255, 255))
    blue_mask = cv2.inRange(hsv_fg, (115, 125, 125), (150, 255, 255))
    bg_mask = cv2.bitwise_or(green_mask, blue_mask)
    open_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    open_mask_inv = cv2.bitwise_not(open_mask)
    img_fg = cv2.bitwise_and(im, im, mask=open_mask_inv)
    img_empty = np.ones_like(img_fg) * 255
    img_bg = cv2.bitwise_and(img_empty, img_empty, mask=open_mask)
    img_fgbg = cv2.add(img_fg, img_bg)
    thresh_im = clean_img(img_fgbg)
    # We use RETR_EXTERNAL mode to find external contour, CHAIN_APPROX_NONE to keep all points without simplification.
    contour = cv2.findContours(thresh_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # Keep the biggest contour only
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    contour_array = contour[0][:, 0, :]
    # create a mask of biggest contour
    color_mask = np.zeros_like(thresh_im)
    cv2.drawContours(color_mask, contour, 0, (255, 255, 255), -1)
    # calculate the average red value of shape
    red = cv2.mean(im, mask=color_mask)[2]
    # Create an array storing the contour points in complex numbers
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:, 0]
    contours_complex.imag = contour_array[:, 1]

    # Compute discrete Fourier Transform and get the amplitude
    fourier_result = np.fft.fft(contours_complex)
    fourier_norm = np.absolute(fourier_result)
    # We abandon the 0's Fourier descriptor because it is variant to positions.
    # We use first descriptor for normalizing all descriptors by the first one
    fourier_norm = fourier_norm / fourier_norm[1]
    isred = 0
    if (red > 100):
        isred = 1
        return fourier_norm[2], fourier_norm[3], isred
    else:
        return fourier_norm[6], fourier_norm[7], isred
    # 2,3 for red set classification
    # 6,7 for black set classification


# The value of bayesian classifier's decision curve is calculated in the notebook Bayesian classifier.ipynb.
# The classifier is a binary classifier works on top of the knowledge of the color of suite image.
def bayes_spade_club(x1, x2):
    if (
            2859.89042987933 * x1 ** 2 - 1233.09652151866 * x1 * x2 + 50.9525608771483 * x1 - 3840.76833708728 * x2 ** 2 + 648.062279494321 * x2 - 38.7202501408227 > 0):
        return 3
    else:
        return 0


def bayes_diamond_heart(x1, x2):
    if (
            -8631.68822729593 * x1 ** 2 + 334.043833410848 * x1 * x2 + 28.8862082239766 * x1 - 401.333957105818 * x2 ** 2 + 743.09053520083 * x2 - 182.895039049464 > 0):
        return 1
    else:
        return 2


# define a downscaling factor for faster pre-processing
rescale_factor = 0.25

# define two structuring elements that will be used in the segmentation part
disk1 = skimage.morphology.disk((int)(35 * rescale_factor))
disk2 = skimage.morphology.disk((int)(13 * rescale_factor))

game_location = "./games/"
games_folder = os.listdir(game_location)
turns = 13
value_classifier_model = './data/value_classifier_256_995_980.mdl'

print("Found these games:{}".format(games_folder))

# Plot boundaries
plot_bound = True

game_count = len(games_folder)
# read data from test set
test_img_raw = []
for game in games_folder:
    # different from training set, we don't have labels this time
    for i in range(turns):
        test_img_raw.append(cv2.imread(game_location + game + '/' + str(i + 1) + '.jpg'))

suit_img_testing = []
value_img_testing = []
dealer_testing = []

print("Processing....")

for img_nb in range(len(test_img_raw)):
    cards, dealer = find_green_boxes(test_img_raw[img_nb])

    contour_dealer, contours_card, M_player, M_dealer, M_suit, suit_img, value_img, dealer_assign \
        = segmentation(cards, dealer, test_img_raw[img_nb])

    for player in range(0, 4):
        suit_img_testing.append(suit_img[player])
        value_img_testing.append(value_img[player])
        dealer_testing.append(dealer_assign[player])


predictor = ValueClassifier(value_classifier_model)
value = predictor.predict(np.array(value_img_testing))

suit = []
for i in range(len(test_img_raw) * 4):
    color = get_fourier_descriptors(suit_img_testing[i])[2]
    if color == 0:  # black
        suit.append(bayes_spade_club(get_fourier_descriptors(suit_img_testing[i])[0],
                                     get_fourier_descriptors(suit_img_testing[i])[1]))
    else:  # red
        suit.append(bayes_diamond_heart(get_fourier_descriptors(suit_img_testing[i])[0],
                                        get_fourier_descriptors(suit_img_testing[i])[1]))

# Arrange the result into

for i in range(game_count):
    turn = []
    for j in range(turns):
        card_matrix = []
        value_turn = []
        suite_turn = []
        dealer_turn = []

        for player_nb in range(0, 4):
            value_turn.append(value[i * turns * 4 + j * 4 + player_nb])
            suite_turn.append(suit[i * turns * 4 + j * 4 + player_nb])
            dealer_turn.append(dealer_testing[i * turns * 4 + j * 4 + player_nb])

        card_matrix.append(value_turn)
        card_matrix.append(suite_turn)
        card_matrix.append(dealer_turn)
        turn.append(np.array(card_matrix, dtype=object))

    pred_rank, pred_dealer, pred_pts_stand, pred_pts_advan = decision(turn)

    print("\n\n==== Result of Game {}====".format(i + 1))
    print_results(
        rank_colour=pred_rank,
        dealer=pred_dealer,
        pts_standard=pred_pts_stand,
        pts_advanced=pred_pts_advan,
    )
