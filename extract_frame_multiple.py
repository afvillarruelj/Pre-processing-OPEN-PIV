# --- CONFIGURATION ---
video_path = "video.mp4" #Load video

real_diameter_m = 0.07  # 7 cm oranges (organic tracers)
N_frames_list = [4, 10, 14, 20, 24, 30, 34, 40, 44, 50, 54, 60, 64, 70, 74]  # <- run your simulations here
#N_frames_list = [80]  # <- run your simulations here


# Crop coordinates (Images)
xmin, ymin, xmax, ymax = 1670, 540, 1850, 720 # case 1: 180x180 pixels
#xmin, ymin, xmax, ymax = 1470, 400, 1970, 900 # case 2: 500x500 pixels
#xmin, ymin, xmax, ymax = 1100, 200, 2400, 1500  # alternative case 3

# HSV range for yellow-orange
lower_yellow = np.array([18, 30, 160])
upper_yellow = np.array([40, 255, 255])

# Acceptable detected diameter range in pixels for the organic tracers (adjust to your case)
diameter_min = 2.5
diameter_max = 3.0

# Get FPS of the video and total frames in the time window
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    raise SystemExit
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"🎞️ Video FPS: {fps:.6f}")

# Time window (adjust if needed)
start_time = 3 * 60 + 35  # 3:35
end_time   = 3 * 60 + 40  # 3:40
start_frame = int((start_time * fps))
end_frame   = int((end_time   * fps))
total_frames_window = end_frame - start_frame
duration_s = (end_frame - start_frame) / fps
print(f"⏱️ Window = {duration_s:.6f} s, frames in window = {total_frames_window}")

cap.release()

# Requested lists
dt_real_list   = []
avg_scale_list = []
results_rows   = []  # for CSV

# ===== LOOP FOR ALL SIMULATIONS =====
for N_frames in N_frames_list:
    print(f"\n🚀 Running simulation with N_frames = {N_frames}")

    # ---- Selection of equally spaced indices in the window ----
    # We use linspace to get EXACTLY N_frames indices between [start_frame, end_frame-1]
    # Note: dtype=int causes quantization to discrete frames (no "half frame")
    indices = np.linspace(start_frame, end_frame - 1, N_frames, dtype=int)

    # Remove duplicates due to rounding (rare with reasonable windows)
    indices = np.unique(indices)
    if len(indices) < N_frames:
        print(f"⚠️ Duplicates removed due to rounding, effective N = {len(indices)}")

    # Average dt_real for this run, measured from selected indices
    if len(indices) >= 2:
        dt_real_run = float(np.mean(np.diff(indices)) / fps)
    else:
        dt_real_run = math.nan

    print(f"✅ Average measured dt_real = {dt_real_run:.6f} s")

    # ---- Folders for this N_frames ----
    frames_folder = f"frames_{N_frames}"
    mask_folder   = f"masks_{N_frames}"
    vis_folder    = f"visual_detections_{N_frames}"
    gauss_folder  = f"a_gaussian_frames_{N_frames}"

    for folder in [frames_folder, mask_folder, vis_folder, gauss_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)

    # === EXTRACT AND SAVE FRAMES ===
    cap = cv2.VideoCapture(video_path)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Could not read frame {idx}")
            continue
        cropped = frame[ymin:ymax, xmin:xmax]
        filename = os.path.join(frames_folder, f"frame_{saved:03d}.png")
        cv2.imwrite(filename, cropped)
        saved += 1
    cap.release()
    print(f"📁 {saved} cropped frames saved in '{frames_folder}'.")

    # === ORANGE DETECTION ===
    image_files = sorted(
        [f for f in os.listdir(frames_folder) if f.endswith(".png")],
        key=lambda x: int(re.findall(r'\d+', x)[0])
    )

    scales_pix_per_m = []
    diams_pix        = []
    detections       = 0
    processed        = 0

    for i, filename in enumerate(image_files):
        img_path = os.path.join(frames_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {filename}")
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.medianBlur(mask, 5)

        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        orange_detected = False

        for cnt in contours:
            ((cx, cy), radius) = cv2.minEnclosingCircle(cnt)
            diameter_pixels = 2 * radius

            if diameter_min <= diameter_pixels <= diameter_max:
                scale = diameter_pixels / real_diameter_m
                scales_pix_per_m.append(scale)
                diams_pix.append(diameter_pixels)
                detections += 1
                orange_detected = True
                # Draw for verification
                cv2.circle(img, (int(cx), int(cy)), int(radius), (0, 255, 0), 2)

        # Save outputs
        cv2.imwrite(os.path.join(mask_folder, f"mask_{i:03d}.png"), mask)
        cv2.imwrite(os.path.join(vis_folder,  f"detect_{i:03d}.png"), img)
        processed += 1

        if orange_detected:
            print(f"✅ {filename}: at least 1 valid orange")
        else:
            print(f"⚠️ {filename}: no fruits within size range")

    # === RESULTS PER RUN ===
    if detections > 0:
        avg_scale = float(sum(scales_pix_per_m) / len(scales_pix_per_m))
        avg_diam  = float(sum(diams_pix) / len(diams_pix))
        print("\n📊 PARTIAL RESULTS")
        print(f"🖼️ Processed images: {processed}")
        print(f"🍊 Valid oranges detected: {detections}")
        print(f"📏 Average scale: {avg_scale:.6f} pix/m")
        print(f"📐 Average diameter: {avg_diam:.3f} px")
    else:
        avg_scale = math.nan
        print("\n📊 PARTIAL RESULTS")
        print(f"🖼️ Processed images: {processed}")
        print("❌ Could not calculate scale — no valid detections")

    print(f"⏱️ dt_real (average measured): {dt_real_run:.6f} s")

    # Save to requested lists
    dt_real_list.append(dt_real_run)
    avg_scale_list.append(avg_scale)

    # === GAUSSIAN PROCESSING ===
    for i, image_file in enumerate(image_files):
        input_path  = os.path.join(frames_folder, image_file)
        output_path = os.path.join(gauss_folder, f"mask_{i:03d}.png")

        img = cv2.imread(input_path)
        if img is None:
            continue
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        for contour in contours:
            cv2.drawContours(mask, [contour], 0, 255, -1)
        result      = cv2.bitwise_and(img, img, mask=mask)
        result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh   = cv2.threshold(result_gray, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite(output_path, thresh)

    print(f"✅ Processed {len(image_files)} images and saved in '{gauss_folder}'.")

    # Register for CSV
    results_rows.append({
        "N_frames": N_frames,
        "dt_real_s": dt_real_run,
        "avg_scale_pix_per_m": avg_scale,
        "detections": detections,
        "processed": processed
    })

# ===== FINAL SUMMARY =====
print("\n================ FINAL SUMMARY ================")
print(f"N_frames used: {N_frames_list}")
print("dt_real_list (s):", [None if (isinstance(v, float) and math.isnan(v)) else round(v, 6) for v in dt_real_list])
print("avg_scale_list (pix/m):", [None if (isinstance(v, float) and math.isnan(v)) else round(v, 6) for v in avg_scale_list])

# Save summary CSV
csv_path = "resumen_runs.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["N_frames", "dt_real_s", "avg_scale_pix_per_m", "detections", "processed"])
    writer.writeheader()
    for row in results_rows:
        writer.writerow(row)

print(f"📝 Summary saved at: {csv_path}")
