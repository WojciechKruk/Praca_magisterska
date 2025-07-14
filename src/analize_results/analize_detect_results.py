import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
import os

# ───────────────────────────────────────
# Parsowanie linii z pliku
# ───────────────────────────────────────

def parse_line(line):
    """Parsuje linijkę zawierającą nazwę pliku, punkt (x, y) i kąt."""
    match = re.match(r"(\S+)\s+\(([\d.]+),\s*([\d.]+)\)\s+([-+]?\d*\.\d+|\d+)", line.strip())
    if not match:
        raise ValueError(f"Niepoprawny format linii: {line}")
    name = match[1]
    x, y = float(match[2]), float(match[3])
    angle = float(match[4])
    return name, (x, y), angle

def parse_line_with_time(line, use_angle=True):
    line = line.strip()

    if use_angle:
        # z kątem: nazwa (x, y) kąt czas
        match = re.match(r"(\S+)\s+\(([\d.]+),\s*([\d.]+)\)\s+([-+]?\d*\.\d+|\d+)\s+([\d.]+)", line)
        if not match:
            raise ValueError(f"Niepoprawny format z czasem i kątem: {line}")
        name = match[1]
        x, y = float(match[2]), float(match[3])
        angle = float(match[4])
        time = float(match[5])
        return name, (x, y), angle, time
    else:
        # bez kąta: nazwa (x, y) czas
        match = re.match(r"(\S+)\s+\(([\d.]+),\s*([\d.]+)\)\s+([\d.]+)", line)
        if not match:
            raise ValueError(f"Niepoprawny format z czasem (bez kąta): {line}")
        name = match[1]
        x, y = float(match[2]), float(match[3])
        angle = 0.0
        time = float(match[4])
        return name, (x, y), time


# ───────────────────────────────────────
# Ładowanie danych
# ───────────────────────────────────────

def load_data(path, with_time=False, use_angle=True):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if with_time:
                if use_angle:
                    name, point, angle, time = parse_line_with_time(line)
                    data[name] = {'point': point, 'angle': angle, 'time': time}
                else:
                    name, point, time = parse_line_with_time(line, use_angle)
                    data[name] = {'point': point, 'time': time}
            else:
                name, point, angle = parse_line(line)
                data[name] = {'point': point, 'angle': angle}
    return data

# ───────────────────────────────────────
# Obliczenia metryk
# ───────────────────────────────────────

def compute_metrics(gt_data, pred_data, use_angle=True):
    distances = []
    angle_diffs = []
    for name, gt in gt_data.items():
        if name not in pred_data:
            continue
        pred = pred_data[name]
        dist = np.linalg.norm(np.array(pred['point']) - np.array(gt['point']))
        distances.append(dist)
        if use_angle:
            # angle_diff = abs((pred['angle'] - gt['angle']))
            angle_diff = abs((pred['angle'] - gt['angle']) % 90)
            if angle_diff > 45:
                angle_diff = 90 - angle_diff
            angle_diffs.append(angle_diff)
    if not use_angle:
        angle_diffs = np.zeros_like(distances)
    return np.array(distances), np.array(angle_diffs)


def count_hits(distances, angle_diffs, use_angle=True, dist_thresh=40, angle_thresh=5):
    if use_angle:
        mask = (distances < dist_thresh) & (angle_diffs < angle_thresh)
    else:
        mask = distances < dist_thresh
    return np.sum(mask)


# ───────────────────────────────────────
# Wizualizacja
# ───────────────────────────────────────

def plot_histogram(data, title, xlabel, bins=30):
    plt.figure()
    plt.hist(data, bins=bins, color='gray', edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Liczba próbek")
    plt.grid(True)
    plt.tight_layout()

def plot_scatter(dist1, angle1, dist2=None, angle2=None, label1="Model CV-X", label2=""):
    plt.figure()
    plt.scatter(dist1, angle1, alpha=0.6, label=label1, color="green", marker='o')
    if dist2 is not None and angle2 is not None:
        plt.scatter(dist2, angle2, alpha=0.6, label=label2, color="blue", marker='x')
    plt.title("Rozrzut różnicy położenia względem referencji względem błędu kąta\n- model CV-X")
    plt.xlabel("Różnica położenia względem punktu referencyjnego [px]")
    plt.ylabel("Różnica kąta względem referencji [°]")
    plt.grid(True)
    plt.tight_layout()

def show_detection_points(name, image_dir, ref_data, my_data, cvx_data):
    import cv2
    import matplotlib.pyplot as plt

    image_path = os.path.join(image_dir, name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Nie znaleziono obrazu: {image_path}")
        return

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    ax = plt.gca()

    # Punkty
    if name in ref_data:
        x, y = ref_data[name]['point']
        ax.plot(x, y, 'bo', label='Referencja')
        ax.text(x+5, y-5, 'Ref', color='blue')
    if name in my_data:
        x, y = my_data[name]['point']
        ax.plot(x, y, 'ro', label='MyModel')
        ax.text(x+5, y-5, 'My', color='red')
    if name in cvx_data:
        x, y = cvx_data[name]['point']
        ax.plot(x, y, 'go', label='CV-X')
        ax.text(x+5, y-5, 'CV-X', color='green')

    plt.title(f"Punkty detekcji – {name}")
    plt.axis('off')
    plt.legend()
    plt.tight_layout()
    plt.show()


# ───────────────────────────────────────
# Main
# ───────────────────────────────────────

def main():
    labels_path = "loc_labels.txt"
    my_path = "My_model_detect_test/My_detect_results.txt"
    cvx_path = "CV-X_detect_test/CV-X_detect_results.txt"

    # Folder docelowy
    out_dir = "detection_results_analysis"
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    reference_data = load_data(labels_path)
    my_data = load_data(my_path, with_time=True, use_angle=False)
    cvx_data = load_data(cvx_path, with_time=True)

    my_dist, my_angle = compute_metrics(reference_data, my_data, use_angle=False)
    cvx_dist, cvx_angle = compute_metrics(reference_data, cvx_data, use_angle=True)

    my_hits = count_hits(my_dist, my_angle, use_angle=False)
    cvx_hits = count_hits(cvx_dist, cvx_angle, use_angle=True)

    print("== Porównanie detekcji ==")
    print(f"MyModel: średnia odległość = {my_dist.mean():.2f}px")
    print(f"CV-X:     średnia odległość = {cvx_dist.mean():.2f}px, średnia różnica kąta = {cvx_angle.mean():.2f}°")
    print(f"MyModel: trafień (odl < 40px): {my_hits} / {len(my_dist)}")
    print(f"CV-X:     trafień (odl < 40px, kąt < 5°): {cvx_hits} / {len(cvx_dist)}")

    # # Wyświetlenie punktów dla wybranego obrazu
    # test_image_name = "img_0016.jpg"
    # image_directory = r"C:\Users\krukw\PycharmProjects\Baumer_test\src\fake data\generated_fake_data_test_detection_dataset"
    #
    # show_detection_points(test_image_name, image_directory, reference_data, my_data, cvx_data)
    #
    # print("\n== Szczegóły CV-X dla pierwszych 100 obrazów ==")
    # sorted_names = sorted(reference_data.keys())[:100]
    # for name in sorted_names:
    #     if name not in cvx_data:
    #         continue
    #     ref_pt = np.array(reference_data[name]["point"])
    #     pred_pt = np.array(cvx_data[name]["point"])
    #     dist = np.linalg.norm(pred_pt - ref_pt)
    #
    #     ref_angle = reference_data[name]["angle"]
    #     pred_angle = cvx_data[name]["angle"]
    #     angle_diff = abs((pred_angle - ref_angle) % 90)
    #     if angle_diff > 45:
    #         angle_diff = 90 - angle_diff
    #
    #     print(f"{name:<20} odległość: {dist:6.2f}px   różnica kąta: {angle_diff:5.2f}°")


    # Eksport CSV
    csv_path = os.path.join(out_dir, "results_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "my_dist", "cvx_dist", "cvx_angle"])
        for name in reference_data:
            my_d = my_data.get(name, {}).get("point", None)
            cvx_d = cvx_data.get(name, {}).get("point", None)
            my_val = np.linalg.norm(np.array(my_d) - np.array(reference_data[name]["point"])) if my_d else ""
            cvx_val = np.linalg.norm(np.array(cvx_d) - np.array(reference_data[name]["point"])) if cvx_d else ""
            cvx_ang = abs((cvx_data.get(name, {}).get("angle", 0) - reference_data[name].get("angle", 0)) % 90)
            # cvx_ang = abs((cvx_data.get(name, {}).get("angle", 0) - reference_data[name].get("angle", 0)))
            if cvx_ang > 45:
                cvx_ang = 90 - cvx_ang
            writer.writerow([name, my_val, cvx_val, cvx_ang])

    # Wykresy
    plot_histogram(cvx_dist, "Histogram różnicy położenia względem punktu referencyjnego\n– model CV-X",
                   "Różnica położenia [px]")
    plt.savefig(os.path.join(plot_dir, "hist_odleglosci_CV-X.png"))

    plot_histogram(my_dist, "Histogram różnicy położenia względem punktu referencyjnego\n– model własny",
                   "Różnica położenia [px]")
    plt.savefig(os.path.join(plot_dir, "hist_odleglosci_My_model.png"))

    plot_histogram(cvx_angle, "Histogram różnicy kąta względem kąta referencyjnego\n– model CV-X", "Różnica kąta [°]")
    plt.savefig(os.path.join(plot_dir, "hist_katow_cvx.png"))

    plot_scatter(cvx_dist, cvx_angle)
    plt.savefig(os.path.join(plot_dir, "scatter_odl_vs_kat.png"))

    # Wykres słupkowy
    plt.figure()
    plt.bar(np.arange(len(my_dist)), my_dist, color="skyblue")
    plt.title("Rozkład różnicy położenia względem referencji\n– model własny")
    plt.xlabel("Nr próbki")
    plt.ylabel("Odległość [px]")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "bar_my_model_odleglosci.png"))

    plt.figure()
    plt.bar(np.arange(len(cvx_dist)), cvx_dist, color="skyblue")
    plt.title("Rozkład różnicy położenia względem referencji\n– model CV-X")
    plt.xlabel("Nr próbki")
    plt.ylabel("Odległość [px]")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "bar_CV-X_odleglosci.png"))

    # Wykres: czas vs. odległość (posortowane)
    # MyModel
    my_names = [name for name in reference_data if name in my_data]
    my_points = [(name, np.linalg.norm(np.array(my_data[name]["point"]) - np.array(reference_data[name]["point"])),
                  my_data[name]["time"])
                 for name in my_names]
    my_points.sort(key=lambda x: x[1])  # sortuj po odległości

    my_dists_sorted = [pt[1] for pt in my_points]
    my_times_sorted = [pt[2] for pt in my_points]

    plt.figure()
    plt.plot(my_dists_sorted, my_times_sorted, marker='o', linestyle='-', color='blue')
    plt.title("Zależność czasu działania od różnicy położenia\n– model własny")
    plt.xlabel("Odległość od referencji [px]")
    plt.ylabel("Czas detekcji [s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "scatter_odl_vs_time_MyModel.png"))

    # CV-X
    cvx_names = [name for name in reference_data if name in cvx_data]
    cvx_points = [(name, np.linalg.norm(np.array(cvx_data[name]["point"]) - np.array(reference_data[name]["point"])),
                   cvx_data[name]["time"])
                  for name in cvx_names]
    cvx_points.sort(key=lambda x: x[1])  # sortuj po odległości

    cvx_dists_sorted = [pt[1] for pt in cvx_points]
    cvx_times_sorted = [pt[2] for pt in cvx_points]

    plt.figure()
    plt.plot(cvx_dists_sorted, cvx_times_sorted, marker='o', linestyle='-', color='green')
    plt.title("Zależność czasu działania od różnicy położenia\n– model CV-X")
    plt.xlabel("Odległość od referencji [px]")
    plt.ylabel("Czas detekcji [s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "scatter_odl_vs_time_CVX.png"))

    # Histogram: średni czas detekcji względem odległości – MyModel
    bin_edges = np.histogram_bin_edges(my_dists_sorted, bins=20)
    bin_indices = np.digitize(my_dists_sorted, bin_edges)
    bin_centers = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    avg_times = [np.mean([my_times_sorted[j] for j in range(len(my_dists_sorted)) if bin_indices[j]==i+1])
                 if any(bin_indices[j]==i+1 for j in range(len(my_dists_sorted))) else 0
                 for i in range(len(bin_centers))]

    plt.figure()
    plt.bar(bin_centers, avg_times, width=np.diff(bin_edges), align="center", color='blue', edgecolor='black', alpha=0.7)
    plt.title("Średni czas detekcji względem różnicy położenia\n– model własny")
    plt.xlabel("Odległość od referencji [px]")
    plt.ylabel("Średni czas detekcji [s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "hist_time_vs_distance_MyModel.png"))

    # Histogram: średni czas detekcji względem odległości – CV-X
    bin_edges = np.histogram_bin_edges(cvx_dists_sorted, bins=20)
    bin_indices = np.digitize(cvx_dists_sorted, bin_edges)
    bin_centers = [(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
    avg_times = [np.mean([cvx_times_sorted[j] for j in range(len(cvx_dists_sorted)) if bin_indices[j]==i+1])
                 if any(bin_indices[j]==i+1 for j in range(len(cvx_dists_sorted))) else 0
                 for i in range(len(bin_centers))]

    plt.figure()
    plt.bar(bin_centers, avg_times, width=np.diff(bin_edges), align="center", color='green', edgecolor='black', alpha=0.7)
    plt.title("Średni czas detekcji względem różnicy położenia\n– model CV-X")
    plt.xlabel("Odległość od referencji [px]")
    plt.ylabel("Średni czas detekcji [s]")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "hist_time_vs_distance_CVX.png"))


    plt.show()

try:
    if __name__ == "__main__":
        main()
except KeyboardInterrupt:
    print("\n🛑 Program został przerwany przez użytkownika (KeyboardInterrupt). Zamykanie...")
