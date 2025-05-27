import cv2
import numpy as np
import networkx as nx
import random
import os
import json
import datetime
import string
from tkinter import Tk, filedialog

def upload_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image to Encrypt",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    return file_path

def get_unique_output_folder(base_dir="./encrypted_output"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    unique_folder = os.path.join(base_dir, f"enc_session_{timestamp}_{rand_suffix}")
    os.makedirs(unique_folder, exist_ok=True)
    return unique_folder

def image_to_graph(image):
    height, width = image.shape
    G = nx.Graph()
    for x in range(height):
        for y in range(width):
            pixel_value = int(image[x, y])
            G.add_node((x, y), intensity=pixel_value)
            if x > 0:
                diff = abs(pixel_value - int(image[x - 1, y]))
                G.add_edge((x, y), (x - 1, y), weight=diff)
            if y > 0:
                diff = abs(pixel_value - int(image[x, y - 1]))
                G.add_edge((x, y), (x, y - 1), weight=diff)
    return G

def generate_dual_mst(G):
    mst1 = nx.minimum_spanning_tree(G, algorithm='kruskal')
    G_random = G.copy()
    for u, v, d in G_random.edges(data=True):
        d['weight'] += random.randint(1, 10)
    mst2 = nx.minimum_spanning_tree(G_random, algorithm='kruskal')
    merged_mst = nx.compose(mst1, mst2)
    return merged_mst

def cluster_pixels(mst):
    clusters = list(nx.connected_components(mst))
    cluster_map = {}
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            cluster_map[node] = cluster_id
    return cluster_map

def shuffle_pixels(image, cluster_map):
    shuffled_image = np.zeros_like(image)
    shuffle_map = {}
    for cluster_id in set(cluster_map.values()):
        cluster_pixels = [pos for pos, cid in cluster_map.items() if cid == cluster_id]
        shuffled_positions = random.sample(cluster_pixels, len(cluster_pixels))
        shuffle_map[cluster_id] = dict(zip(cluster_pixels, shuffled_positions))
        for original, shuffled in shuffle_map[cluster_id].items():
            shuffled_image[shuffled] = image[original]
    return shuffled_image, shuffle_map

def encrypt_image(image, mst):
    encrypted_image = image.copy()
    for (u, v, data) in mst.edges(data=True):
        x1, y1 = u
        weight = int(data['weight']) % 256
        encrypted_image[x1, y1] ^= weight
    return encrypted_image

def save_encryption_data(output_dir, encrypted_image, shuffle_map, cluster_map, mst, base_name):
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_encrypted.png"), encrypted_image)

    with open(os.path.join(output_dir, f"{base_name}_shuffle_map.json"), 'w') as f:
        json.dump({str(k): {str(k2): str(v2) for k2, v2 in v.items()} for k, v in shuffle_map.items()}, f, indent=2)

    with open(os.path.join(output_dir, f"{base_name}_cluster_map.json"), 'w') as f:
        json.dump({str(k): int(v) for k, v in cluster_map.items()}, f, indent=2)

    mst_edges = [{"from": str(u), "to": str(v), "weight": int(d["weight"])} for u, v, d in mst.edges(data=True)]
    with open(os.path.join(output_dir, f"{base_name}_mst_edges.json"), 'w') as f:
        json.dump(mst_edges, f, indent=2)

def main():
    image_path = upload_image()
    if not image_path:
        print("❌ No image selected.")
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = get_unique_output_folder()
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    G = image_to_graph(image)
    mst = generate_dual_mst(G)
    cluster_map = cluster_pixels(mst)
    shuffled_image, shuffle_map = shuffle_pixels(image, cluster_map)
    encrypted_image = encrypt_image(shuffled_image, mst)

    save_encryption_data(output_dir, encrypted_image, shuffle_map, cluster_map, mst, base_name)
    print(f"[✓] Encryption complete. Files saved to {output_dir}")

if __name__ == "__main__":
    main()
