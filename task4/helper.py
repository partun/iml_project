import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
from skimage import transform
from sklearn.model_selection import train_test_split

seed_value = 43
np.random.seed(seed_value)


def gen_path_files(filepath, datapath, p, image_count=10000):
    triplets = np.genfromtxt(filepath, dtype=np.int)

    triplets_flat = triplets.flatten()
    im_occurance = np.zeros((image_count,))
    all_im = np.arange(image_count)

    for i in range(len(triplets_flat)):
        im_occurance[triplets_flat[i]] += 1

    all_im = all_im[im_occurance != 0]
    train_im, val_im = train_test_split(all_im, test_size=p)

    train_output = open('path_train_triplets.txt', 'w')
    val_output = open('path_val_triplets.txt', 'w')
    val_c = 0
    train_c = 0
    total_triplet_c = triplets.shape[0]

    for i in range(total_triplet_c):
        if triplets[i, 0] in val_im and triplets[i, 1] in val_im and triplets[i, 2] in val_im:
            val_output.write('{}{:05d}.jpg,{}{:05d}.jpg,{}{:05d}.jpg\n'.format(
                datapath, triplets[i, 0], datapath, triplets[i, 1], datapath, triplets[i, 2]))
            val_c += 1
        elif triplets[i, 0] in train_im and triplets[i, 1] in train_im and triplets[i, 2] in train_im:
            train_output.write('{}{:05d}.jpg,{}{:05d}.jpg,{}{:05d}.jpg\n'.format(
                datapath, triplets[i, 0], datapath, triplets[i, 1], datapath, triplets[i, 2]))
            train_c += 1

    lost_triplet_c = total_triplet_c - train_c - val_c

    print('=================== train validation split ==================================================================')
    print('Total Triplets: {}   Train Triplets: {}  Val Triplets: {} Lost Triplets: {} ({:01f}%)'.format(
        total_triplet_c, train_c, val_c, lost_triplet_c, lost_triplet_c / total_triplet_c * 100))
    max_div = commen_divisors(train_c, val_c)
    print('==============================================================================================================')

    val_output.close()
    train_output.close()
    return max_div, train_c, val_c


def commen_divisors(x, y, max_div=15):
    divs = []

    for d in range(1, min(x, y) // 2):
        if x % d == 0 and y % d == 0:
            divs.append(d)

    max_div = max(divs, key=lambda x: x if x <= max_div else 0)
    print('Possible Divisors: {} Selected Divisor: {}'.format(str(divs), max_div))
    return max_div


class prediction_generator:

    def __init__(self, image_path, deep_rank_model, nm=False):
        self.image_path = image_path
        self.deep_rank_model = deep_rank_model
        self.nm = nm

    def id_to_path(self, i):
        # transforms 1 to 'data/food/00001.jpg'
        return self.image_path + "{:05d}".format(i) + ".jpg"

    def get_image_score(self, im_id):
        # calculates the output of the nn for im_id
        path = self.id_to_path(im_id)
        image = load_img(path)
        image = img_to_array(image).astype('float64')
        image = transform.resize(image, (224, 224))
        image *= 1./255
        image = np.expand_dims(image, axis=0)

        if self.nm:
            return self.deep_rank_model.predict(image)[0]
        else:
            return self.deep_rank_model.predict([image, image, image])[0]

    def pair(self, i, j):
        # make tuple with the lower number first
        # used as key in the distance cache

        if i < j:
            return (i, j)
        else:
            return (j, i)

    def generate_output(self, path, out_path='output.txt'):
        # generates prodiction for triplets in path file

        score = dict()
        distance_cache = dict()
        for i in tqdm(range(10000)):
            score[i] = self.get_image_score(i)

        test_triplets = np.genfromtxt(
            'test_triplets.txt', dtype=np.int, delimiter=' ')
        output = open(out_path, 'w')

        for i in tqdm(range(test_triplets.shape[0])):
            try:
                q = test_triplets[i, 0]
                p = test_triplets[i, 1]
                n = test_triplets[i, 2]

                score_q = None
                score_p = None
                score_n = None

                qp = self.pair(q, p)
                if qp in distance_cache:
                    distance_p = distance_cache[qp]
                else:
                    score_q = score[test_triplets[i, 0]]
                    score_p = score[test_triplets[i, 1]]
                    distance_p = sum([(score_q[idx] - score_p[idx]) **
                                      2 for idx in range(len(score_q))])

                distance_cache[qp] = distance_p

                qn = self.pair(q, n)
                if qn in distance_cache:
                    distance_n = distance_cache[qn]
                else:
                    if score_q is None:
                        score_q = score[test_triplets[i, 0]]
                    score_n = score[test_triplets[i, 2]]
                    distance_n = sum([(score_q[idx] - score_n[idx]) **
                                      2 for idx in range(len(score_q))])

                    distance_cache[qn] = distance_n

            except:
                KeyError
                distance_n = 0
                distance_p = 0
                print('key not found! ' + str(i))

            if distance_p < distance_n:
                output.write('1\n')
            else:
                output.write('0\n')

            if i % 1000 == 0:
                output.flush()

        output.close()
