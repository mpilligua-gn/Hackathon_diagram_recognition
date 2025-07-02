import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import KMeans
import math
import random
from predict import predict_mode
from units import geometry
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def get_shapes(bbox, cls):
    ret = list()
    for i, x in enumerate(cls):
        if x >= 8:
            continue
        tmp = bbox[i].tolist()
        tmp.append(x)
        ret.append(tmp)
    return ret


def get_edges(bbox, cls):
    ret = list()
    for i, x in enumerate(cls):
        if x <= 8:
            continue
        tmp = bbox[i].tolist()
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
        # Calculate the index of the key point on the shape that has the minimum distance to y.
        dis, direction = geometry.calc(x, y, shape)
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


def normalize_points(points, image_size=1000, padding=20):
    strokes = [np.array(stroke) for stroke in points]
    # get min and max points
    pts = np.concatenate(strokes, axis=0)

    # Get bounds
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y

    # Avoid division by zero
    if width == 0:
        width = 1
    if height == 0:
        height = 1

    # Scale to fit inside (image_size - padding*2)
    scale = min((image_size - 2 * padding) / width,
                (image_size - 2 * padding) / height)

    if scale > 1:
        # If the scale is greater than 1, we can use the original size
        scale = 1

    normalized = []
    for stroke in strokes:
        norm_stroke = [
            ((np.array(p) - np.array([min_x, min_y])) * scale + padding).tolist()
            for p in stroke
        ]
        normalized.append(norm_stroke)

    return normalized, min_x, min_y, scale, padding


def draw_handwriting(norm_points, image_size=1000, save_path="handwriting.png"):

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)

    # Draw lines between points
    for stroke in norm_points:
        if len(stroke) < 2:
            continue
        for i in range(len(stroke) - 1):
            draw.line([tuple(stroke[i]), tuple(stroke[i + 1])], fill="black", width=2)

    img.save(save_path)
    return save_path


def infer_from_handwriting_points(handwriting):
    """
    :param img_path: image path
    :return: pred, edge,
    """
    norm_handwriting, min_x, min_y, scale, padding = normalize_points(handwriting, image_size=1000)
    image_path = draw_handwriting(norm_handwriting, image_size=1000)
    shapes, edges = infer_flowmind2digital(image_path)
    # Adjust shapes and edges based on the handwriting points
    for shape in shapes:
        shape[0] = (shape[0] - padding)/scale
        shape[1] = (shape[1] - padding)/scale
        shape[2] = (shape[2] - padding)/scale
        shape[3] = (shape[3] - padding)/scale

        shape[0] = (shape[0] + min_x)
        shape[1] = (shape[1] + min_y)
        shape[2] = (shape[2] + min_x)
        shape[3] = (shape[3] + min_y)
    for edge in edges:
        edge[0] = (edge[0] - padding)/scale
        edge[1] = (edge[1] - padding)/scale
        edge[2] = (edge[2] - padding)/scale
        edge[3] = (edge[3] - padding)/scale

        edge[0] = (edge[0] + min_x)
        edge[1] = (edge[1] + min_y)
        edge[2] = (edge[2] + min_x)
        edge[3] = (edge[3] + min_y)

    image_path = draw_handwriting(
        handwriting, image_size=1000, save_path='handwriting_result.png')
    draw_on_image('handwriting_shape_result.png', image_path, shapes, edges)
    return shapes, edges


def infer_flowmind2digital(img_path):
    """
    :param img_path: image path
    :return: pred, edge,
    """
    print("Infer flowmind2digital from image:", img_path)
    bbox, cls, kpt, siz = model(img_path=img_path, opt=0)  # pre0/ dataset1

    shapes = get_shapes(bbox, cls)  # get autoshape's bbox and cls
    edges = get_edges(bbox, cls)  # get connector's bbox and cls
    shapes = to_python_floats(shapes)
    edges = to_python_floats(edges)
    return shapes, edges


def to_python_floats(matrix):
    return [[float(x) for x in row] for row in matrix]


# Shape mapping
shape_map = {
    0: 'circle', 1: 'diamonds', 2: 'long_oval', 3: 'hexagon',
    4: 'parallelogram', 5: 'rectangle', 6: 'trapezoid', 7: 'triangle',
    8: 'text', 9: 'arrow', 10: 'double_arrow', 11: 'line'


}


def to_inches(val):
    return val / 96.0


def draw_shape(ax, shape_type, center, width, height):
    cx, cy = center
    if shape_type == 'circle':
        ax.add_patch(mpatches.Circle((cx, cy), radius=width/2, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'diamonds':
        ax.add_patch(mpatches.RegularPolygon((cx, cy), numVertices=4, radius=width/np.sqrt(2),
                     orientation=np.pi/4, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'long_oval':
        ax.add_patch(mpatches.Ellipse((cx, cy), width=width, height=height, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'hexagon':
        ax.add_patch(mpatches.RegularPolygon((cx, cy), numVertices=6,
                     radius=width/2, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'parallelogram':
        ax.add_patch(mpatches.Polygon([[cx - width/2, cy - height/2],
                                       [cx + width/2, cy - height/2],
                                       [cx + width/2 - width/4, cy + height/2],
                                       [cx - width/2 - width/4, cy + height/2]],
                                      closed=True, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'rectangle':
        ax.add_patch(mpatches.Rectangle((cx - width/2, cy - height/2), width,
                     height, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'trapezoid':
        ax.add_patch(mpatches.Polygon([[cx - width/2 + width/4, cy - height/2],
                                       [cx + width/2 - width/4, cy - height/2],
                                       [cx + width/2, cy + height/2],
                                       [cx - width/2, cy + height/2]],
                                      closed=True, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'triangle':
        ax.add_patch(mpatches.RegularPolygon((cx, cy), numVertices=3, radius=width /
                     2, orientation=np.pi, edgecolor='blue', facecolor='none', lw=2))
    elif shape_type == 'text':
        ax.text(cx, cy, 'Text', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', edgecolor='blue'))
    elif shape_type == 'arrow':
        ax.annotate('', xy=(cx + width/2, cy), xytext=(cx - width/2, cy),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    elif shape_type == 'double_arrow':
        ax.annotate('', xy=(cx + width/2, cy), xytext=(cx - width/2, cy),
                    arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    elif shape_type == 'line':
        ax.plot([cx - width/2, cx + width/2], [cy, cy], color='blue', lw=2)


def draw_on_image(path, image_path, shapes, edges):
    # Load image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    centers = []

    # Draw shapes
    for i, box in enumerate(shapes):
        x0, y0 = box[0], box[1]
        x1, y1 = box[2], box[3]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        shape_type = shape_map.get(int(box[4]), 'rectangle')

        if shape_type == 'rectangle':
            draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)
        elif shape_type == 'circle':
            draw.ellipse([x0, y0, x1, y1], outline="blue", width=2)
        elif shape_type == 'long_oval':
            draw.ellipse([x0, y0, x1, y1], outline="blue", width=2)
        elif shape_type == 'diamonds':
            draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], outline="blue")
        elif shape_type == 'hexagon':
            w, h = x1 - x0, y1 - y0
            draw.polygon([
                (cx - w / 2, cy),
                (cx - w / 4, y0),
                (cx + w / 4, y0),
                (cx + w / 2, cy),
                (cx + w / 4, y1),
                (cx - w / 4, y1),
            ], outline="blue")
        elif shape_type == 'triangle':
            draw.polygon([(cx, y0), (x1, y1), (x0, y1)], outline="blue")
        else:
            draw.rectangle([x0, y0, x1, y1], outline="blue", width=2)

        draw.text((cx, cy), str(i), fill="red", font=font, anchor="mm")
        centers.append((cx, cy))

    # Draw connectors as lines with arrowheads
    for i, box in enumerate(edges):
        x0, y0 = box[0], box[1]
        x1, y1 = box[2], box[3]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    img.save(path)
    return path


if __name__ == "__main__":
    img_path = "/home/anh/diagram_recognition/Hackathon_diagram_recognition/flow2digital_poetry/test_1.jpg"  # Path to your image
    shapes, edges = infer_flowmind2digital(img_path)
    draw_on_image(
        "/home/anh/diagram_recognition/Hackathon_diagram_recognition/flow2digital_poetry/test_result.png", img_path, shapes, edges)
    print("Inference completed")
    print("Shapes:", shapes)
    print("Edges:", edges)
