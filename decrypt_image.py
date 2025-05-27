import cv2
import numpy as np
import networkx as nx
import json
import os
from tkinter import Tk, filedialog
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def select_file(title, filetypes):
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=filetypes)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_mst_from_json(edge_list):
    G = nx.Graph()
    for edge in edge_list:
        u = tuple(map(int, edge["from"].strip("()").split(", ")))
        v = tuple(map(int, edge["to"].strip("()").split(", ")))
        weight = edge["weight"]
        G.add_edge(u, v, weight=weight)
    return G

def decrypt_image(encrypted_image, mst):
    decrypted_image = encrypted_image.copy()
    for (u, v, data) in mst.edges(data=True):
        x1, y1 = u
        weight = int(data['weight']) % 256
        decrypted_image[x1, y1] ^= weight
    return decrypted_image

def unshuffle_image(shuffled_image, cluster_map, shuffle_map):
    unshuffled = np.zeros_like(shuffled_image)
    for cluster_id, mapping in shuffle_map.items():
        inverse = {tuple(map(int, v.strip("()").split(", "))): tuple(map(int, k.strip("()").split(", ")))
                   for k, v in mapping.items()}
        for shuffled, original in inverse.items():
            unshuffled[original] = shuffled_image[shuffled]
    return unshuffled

def main():
    encrypted_path = select_file("Select Encrypted Image", [("PNG files", "*.png")])
    shuffle_path = select_file("Select Shuffle Map JSON", [("JSON files", "*.json")])
    cluster_path = select_file("Select Cluster Map JSON", [("JSON files", "*.json")])
    mst_path = select_file("Select MST Edges JSON", [("JSON files", "*.json")])

    if not (encrypted_path and shuffle_path and cluster_path and mst_path):
        print("❌ Missing one or more input files.")
        return

    encrypted_image = cv2.imread(encrypted_path, cv2.IMREAD_GRAYSCALE)
    shuffle_map = load_json(shuffle_path)
    cluster_map = {tuple(map(int, k.strip("()").split(", "))): v for k, v in load_json(cluster_path).items()}
    mst = load_mst_from_json(load_json(mst_path))

    unshuffled_image = unshuffle_image(encrypted_image, cluster_map, shuffle_map)
    decrypted_image = decrypt_image(unshuffled_image, mst)

    save_path = os.path.splitext(encrypted_path)[0] + "_decrypted.png"
    cv2.imwrite(save_path, decrypted_image)
    print(f"[✓] Decryption complete. Decrypted image saved as {save_path}")

    compare = input("Do you want to compare with the original image? (y/n): ").lower()
    if compare == 'y':
        original_path = select_file("Select Original Image", [("Image files", "*.png;*.jpg;*.jpeg")])
        if original_path:
            original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            psnr_val = psnr(original, decrypted_image)
            ssim_val = ssim(original, decrypted_image)
            print(f"📊 PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
        else:
            print("Skipped comparison.")

if __name__ == "__main__":
    main()
