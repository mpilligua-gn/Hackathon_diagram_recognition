import numpy as np
from sklearn.cluster import KMeans
import math
import random
from predict import predict_mode
from units import geometry

class Canopy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.t1 = 0
        self.t2 = 0

    # Set the initial threshold.
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')

    @staticmethod
    def euclideanDistance(vec1, vec2):
        return math.sqrt(((vec1 - vec2)**2).sum())

    # Randomly select an index based on the current length of the dataset.
    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)

    def clustering(self):
        canopies = []  # Container for storing the final clustering results.
        while len(self.dataset) > 1:
            rand_index = self.getRandIndex()
            current_center = self.dataset[rand_index]  # Randomly select a center point and assign it to point P.
            current_center_list = []  # Initialize the canopy category container for point P.
            delete_list = []  # Initialize the deletion container for point P.
            self.dataset = np.delete(self.dataset, rand_index,
                                     0)  # Remove the randomly selected center point P.
            for datum_j in range(len(self.dataset)):
                datum = self.dataset[datum_j]
                distance = self.euclideanDistance(
                    current_center, datum)  # Calculate the distance between the selected center point P and each point.
                if distance < self.t1:
                    #  If the distance is less than t1, assign the point to the canopy category of point P.
                    current_center_list.append(datum)
                if distance < self.t2:
                    delete_list.append(datum_j)  # Add to the deletion container if the value is less than t2.
            # Remove elements from the dataset based on the indices in the deletion container.
            self.dataset = np.delete(self.dataset, delete_list, 0)
            canopies.append((current_center, current_center_list))

        k = len(canopies)
        if len(self.dataset) == 1:
            k += 1
        return k

def clustering(X, t1=1.5, t2=0.5, dim=1):
    """
    :param: Cluster the instances in one-dimensional feature vector X that have a value less than t2 into one category (NumPy or a list).
    :return:  result X
    """
    X = np.array(X)
    X = X.reshape(-1, dim)
    gc = Canopy(X)
    gc.setThreshold(t1, t2)
    k = gc.clustering()
    print("t2: ", t2, "k: ", k)
    if k == 1:
        Y = np.zeros(len(X), dtype='int32')
    else:
        Y = KMeans(n_clusters=k).fit_predict(X)
    avg = np.zeros((k, dim))  # Mean value for each category.
    cnt = np.zeros((k, dim))  # Number of instances for each category.
    for x, y in zip(X, Y):
        avg[y] += x
        cnt[y] += 1
    avg = avg / cnt
    ret = np.zeros_like(X)
    for i, y in enumerate(Y):
        ret[i] = avg[y]
    return ret

def align(pred):
    """
    :param pred: boundingbox
    :return:   Bounding boxes after clustering.
    """
    pred = np.array(pred)  # Half of the mean length is used as t2.
    tx = 1e18
    ty = 1e18
    for box in pred:
        tx = min(box[2] - box[0], tx)
        ty = min(box[3] - box[1], ty)
    tx /= 1.618
    ty /= 1.618

    for i in range(4):
        x = pred[:, i]
        if i & 1:
            x = clustering(x, t2=ty)
        else:
            x = clustering(x, t2=tx)
        pred[:, i] = x.reshape(-1)
    return pred.tolist()

def adjust_shape(pred):
    X = np.zeros((len(pred), 2))
    t2 = 1e18
    for i, box in enumerate(pred):
        X[i][0] = box[2] - box[0]
        X[i][1] = box[3] - box[1]
        t2 = min(t2, math.sqrt(X[i][0]**2 + X[i][1]**2))
    X = clustering(X, dim=2, t2=t2/1.618)
    # print(X)

    for i, box in enumerate(pred):
        midx = (box[2] + box[0]) / 2
        box[0] = midx - X[i][0] / 2
        box[2] = midx + X[i][0] / 2

        midy = (box[3] + box[1]) / 2
        box[1] = midy - X[i][1] / 2
        box[3] = midy + X[i][1] / 2
    return pred

def model(img_path, opt=0):
    """
    :param img_path:
    :param opt: 0 predict / 1 label
    :return: bbox:(num * 4), np
             cls:(num * 1), np
             keypoint:(num * 2 * 2), np
    """
    predict_util = predict_mode()
    arrow_metadata = predict_util.dataset_register(dataset_path=['./dataset_arrow', './dataset_arrow'])
    predict_output = predict_util.predict_flowchart(arrow_metadata=arrow_metadata,
                                                    img_path=img_path,
                                                    save_path='./output/eval.jpg')
    predict_output = predict_output['instances']

    bbox = predict_output.pred_boxes.tensor
    cls = predict_output.pred_classes
    kpt = predict_output.pred_keypoints

    bbox = bbox.cpu().numpy().astype('int32')
    cls = cls.cpu().numpy().astype('int32')
    kpt = kpt.cpu().numpy().astype('int32')
    kpt = kpt[:, :, :-1]
    siz = predict_output.image_size
    return bbox, cls, kpt, siz

def get_pred(bbox, cls):
    ret = list()
    for i, x in enumerate(cls):
        if x >= 8:
            continue
        tmp = bbox[i].tolist()
        tmp.append(x)
        ret.append(tmp)
    return ret

def get_edge(kpt, cls):
    ret = list()
    for i, x in enumerate(cls):
        """
        'arrow': 9,
        'double_arrow': 10,
        'line': 11
        """
        if x < 9:
            continue
        tmp = np.reshape(kpt[i], -1)
        tmp = tmp.tolist()
        tmp.append(x)
        ret.append(tmp)
    return ret

def find_closest_shape(pred, x, y):
    """
    :param pred: all autoshape[xmin, ymin, xmax, ymax, cls]
    :param x:
    :param y:
    :return: opt, id, direction
    """
    mxdis = 1e18
    op = -1
    sp = -1
    d = -1
    for i, shape in enumerate(pred):
        dis, direction = geometry.calc(x, y, shape)  # Calculate the index of the key point on the shape that has the minimum distance to y.
        if dis < mxdis:
            mxdis = dis
            op = 1   # """Set the threshold for the minimum distance."""
            sp = i
            d = direction
    return op, sp, d

def build_graph(pred, edge):
    """
    :param pred: [[x, y, x, y, cls], ...]
    :param edge: [[x, y, x, y, cls], ...]
    :return: an edge graph[op1, id1, 0-3direction, op2, id2, 0-3direction, edge_cls]
             if opt = -1, use original [x, y] instead of id direction
    """
    ret = list()
    for e in edge:
        op1, sp1, d1 = find_closest_shape(pred, e[0], e[1])
        if op1 == -1:
            sp1, d1 = e[0], e[1]
        op2, sp2, d2 = find_closest_shape(pred, e[2], e[3])
        if op2 == -1:
            sp2, d2 = e[2], e[3]
        cur = [op1, sp1, d1,
               op2, sp2, d2,
               e[4]]
        ret.append(cur)
    return ret

def ensure_not_empty_bb(pred):
    """
    :param pred: [[xmin, ymin, xmax, ymax, cls], ...]
    :return: pred with no empty bbox
    """

    for box in pred:
        if box[2] - box[0] < 0:
            box[2] = box[0] + 1
        if box[3] - box[1] < 0:
            box[3] = box[1] + 1 
    return pred

def infer_flowmind2digital(img_path):
    """
    :param img_path: image path
    :return: pred, edge,
    """
    print("Predicting diagram from image:", img_path)
    bbox, cls, kpt, siz = model(img_path=img_path, opt=0)  # pre0/ dataset1

    pred = get_pred(bbox, cls)  # get autoshape's bbox and cls
    edge = get_edge(kpt, cls)  # get connector's bbox and cls

    edge = build_graph(pred, edge)  # build edge
    
    pred = align(pred)
    pred = adjust_shape(pred)  # resizing

    pred = to_python_floats(pred)  # convert to python float for visualization
    edge = to_python_floats(edge)  # convert to python float for visualization

    return pred, edge

def to_python_floats(matrix):
    return [[float(x) for x in row] for row in matrix]

if __name__ == "__main__":
    img_path = "/Users/maria/projects/hackathon_diagram_recognition/flowmind2digital/data/handwritten-diagram-datasets/datasets/fca/test/writer5_4b.png"  # Path to your image
    pred, edge = infer_flowmind2digital(img_path)
    print("Inference completed")