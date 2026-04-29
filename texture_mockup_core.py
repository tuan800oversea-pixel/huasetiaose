import os
import time
from pathlib import Path

import cv2
import numpy as np
import packbits
import pytoshop
import pytoshop.codecs
from pytoshop.enums import BlendMode
from pytoshop.user import nested_layers
from skimage import color
from sklearn.cluster import KMeans


pytoshop.codecs.packbits = packbits


def read_image_robust(image_path, flags=cv2.IMREAD_COLOR):
    """兼容 Windows 中文路径的读取方式。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def save_uploaded_file(uploaded_file, target_dir: str) -> str:
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def get_dynamic_params(pattern_bgr):
    small_pattern = cv2.resize(pattern_bgr, (100, 100))
    pixels = small_pattern.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=5).fit(pixels)
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    bg_color_bgr = kmeans.cluster_centers_[np.argmax(counts)]

    bg_luminance = (
        0.114 * bg_color_bgr[0] + 0.587 * bg_color_bgr[1] + 0.299 * bg_color_bgr[2]
    ) / 255.0

    x_brightness = [0.1, 0.5, 0.9]
    y_gamma = [0.4, 0.6, 0.8]
    y_clip = [0.40, 0.55, 0.75]
    y_alpha = [0.15, 0.08, 0.02]

    gamma_val = float(np.interp(bg_luminance, x_brightness, y_gamma))
    clip_min = float(np.interp(bg_luminance, x_brightness, y_clip))
    alpha_thresh = float(np.interp(bg_luminance, x_brightness, y_alpha))
    return gamma_val, clip_min, alpha_thresh, float(bg_luminance)


def preprocess_mask(mask_path, target_shape):
    mask_raw = read_image_robust(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_raw is None:
        raise FileNotFoundError(f"无法读取蒙版: {mask_path}")

    if len(mask_raw.shape) == 3 and mask_raw.shape[2] == 4:
        mask_gray = mask_raw[:, :, 3]
    else:
        mask_gray = (
            cv2.cvtColor(mask_raw, cv2.COLOR_BGR2GRAY)
            if len(mask_raw.shape) == 3
            else mask_raw
        )
        _, mask_gray = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)

    if mask_gray[0, 0] > 127:
        mask_gray = cv2.bitwise_not(mask_gray)
    if mask_gray.shape[:2] != target_shape:
        mask_gray = cv2.resize(mask_gray, (target_shape[1], target_shape[0]))
    mask_gray = cv2.GaussianBlur(mask_gray, (3, 3), 0)
    return np.repeat((mask_gray.astype(np.float32) / 255.0)[:, :, np.newaxis], 3, axis=2)


def extract_lighting_map(base_bgr, mask_3d, gamma_val, clip_min):
    gray = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2GRAY)
    mask_bool = mask_3d[:, :, 0] > 0.5
    clothing_pixels = gray[mask_bool]

    if len(clothing_pixels) == 0:
        return np.ones_like(base_bgr, dtype=np.float32)

    min_val = np.percentile(clothing_pixels, 2)
    max_val = np.percentile(clothing_pixels, 98)
    lighting = (gray.astype(np.float32) - min_val) / (max_val - min_val + 1e-5)
    lighting = np.clip(lighting, 0, 1)
    lighting = np.power(lighting, gamma_val)
    lighting = lighting * (1.0 - clip_min) + clip_min
    return np.repeat(lighting[:, :, np.newaxis], 3, axis=2)


def create_highlight_overlay_map(lighting_map, mask_3d):
    """Build an Overlay-safe highlight layer with 50% gray as neutral."""
    lighting_gray = lighting_map[:, :, 0]
    highlight_strength = np.clip((lighting_gray - 0.55) / 0.45, 0.0, 1.0)
    highlight_strength = np.power(highlight_strength, 0.8)

    neutral_gray = np.full_like(highlight_strength, 0.5)
    overlay_gray = neutral_gray + highlight_strength * 0.5
    overlay_gray = (
        overlay_gray * mask_3d[:, :, 0] + neutral_gray * (1.0 - mask_3d[:, :, 0])
    )
    return np.repeat(overlay_gray[:, :, np.newaxis], 3, axis=2)


def generate_tiled_texture(
    texture_f,
    garment_h_w,
    scale_factor=1.0,
    offset_x_percent=0,
    offset_y_percent=0,
):
    target_h, target_w = garment_h_w
    tex_h, tex_w = texture_f.shape[:2]
    pattern_size_base = target_w / 2.0
    new_tex_size = pattern_size_base * scale_factor
    scale_y = new_tex_size / tex_h
    scale_x = new_tex_size / tex_w
    resized_base = cv2.resize(
        texture_f, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_CUBIC
    )

    new_w, new_h = resized_base.shape[1], resized_base.shape[0]
    tiles_y = int(np.ceil(target_h / new_h))
    tiles_x = int(np.ceil(target_w / new_w))
    tiled = np.tile(resized_base, (tiles_y, tiles_x, 1))

    max_shift_x = max(0, (tiled.shape[1] - target_w) // 2)
    max_shift_y = max(0, (tiled.shape[0] - target_h) // 2)
    shift_x = int(max_shift_x * (offset_x_percent / 100.0))
    shift_y = int(max_shift_y * (offset_y_percent / 100.0))

    start_y = (tiled.shape[0] - target_h) // 2 + shift_y
    start_x = (tiled.shape[1] - target_w) // 2 + shift_x
    start_y = int(np.clip(start_y, 0, tiled.shape[0] - target_h))
    start_x = int(np.clip(start_x, 0, tiled.shape[1] - target_w))
    return tiled[start_y:start_y + target_h, start_x:start_x + target_w]


def adjust_hsv_image(img_bgr, mask_3d, hue_shift=0, sat_scale=1.0, val_scale=1.0):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    h = np.mod(h + hue_shift, 180)
    s = np.clip(s * sat_scale, 0, 255)
    v = np.clip(v * val_scale, 0, 255)
    hsv_adjusted = cv2.merge([h, s, v]).astype(np.uint8)
    bgr_adjusted = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)

    orig_f = img_bgr.astype(np.float32) / 255.0
    adj_f = bgr_adjusted.astype(np.float32) / 255.0
    final_f = adj_f * mask_3d + orig_f * (1.0 - mask_3d)
    return (np.clip(final_f, 0, 1) * 255.0).astype(np.uint8)


def apply_pattern_and_color_adjust(
    base_black_bgr,
    mask_3d,
    lighting_map,
    pattern_tile_f,
    alpha_thresh,
    scale_factor=1.0,
    offset_x_percent=0,
    offset_y_percent=0,
    hue_shift=0,
    sat_scale=1.0,
    val_scale=1.0,
):
    h, w = base_black_bgr.shape[:2]
    base_f = base_black_bgr.astype(np.float32) / 255.0
    tiled_f = generate_tiled_texture(
        pattern_tile_f,
        (h, w),
        scale_factor,
        offset_x_percent,
        offset_y_percent,
    )

    tiled_gray = cv2.cvtColor(
        (np.clip(tiled_f, 0, 1) * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).astype(np.float32) / 255.0
    pattern_alpha = np.clip((tiled_gray - alpha_thresh) * 10.0, 0, 1)
    pattern_alpha_3d = np.repeat(pattern_alpha[:, :, np.newaxis], 3, axis=2)

    pattern_with_light = tiled_f * lighting_map
    blended_f = pattern_with_light * pattern_alpha_3d + base_f * (1.0 - pattern_alpha_3d)
    patterned_mockup_f = blended_f * mask_3d + base_f * (1.0 - mask_3d)

    return adjust_hsv_image(
        (np.clip(patterned_mockup_f, 0, 1) * 255.0).astype(np.uint8),
        mask_3d,
        hue_shift,
        sat_scale,
        val_scale,
    )


def get_color_palette(img_bgr, k=4, crop_center=True):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    if crop_center:
        h, w = img_rgb.shape[:2]
        img_rgb = img_rgb[int(h * 0.3):int(h * 0.8), int(w * 0.3):int(w * 0.7)]

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 2] > 40) & (hsv[:, :, 1] > 30)
    pixels = img_rgb[mask]

    if len(pixels) < k * 10:
        pixels = img_rgb.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    weights = counts / counts.sum()
    return kmeans.cluster_centers_, weights


def calculate_palette_delta_e(target_palette, target_weights, generated_palette):
    total_de = 0.0
    for t_color, t_weight in zip(target_palette, target_weights):
        min_de = float("inf")
        for g_color in generated_palette:
            lab_t = color.rgb2lab(np.array([[t_color]]) / 255.0)
            lab_g = color.rgb2lab(np.array([[g_color]]) / 255.0)
            de = color.deltaE_ciede2000(lab_t, lab_g)[0][0]
            if de < min_de:
                min_de = de
        total_de += min_de * t_weight
    return total_de


def export_layered_psd(
    output_path,
    orig_bgr,
    mask_3d,
    lighting_map,
    pattern_tile_f,
    alpha_thresh,
    best_scale,
    offset_x_percent,
    offset_y_percent,
    best_h,
    best_s,
    best_v,
):
    h, w = orig_bgr.shape[:2]
    alpha_full = np.ascontiguousarray(np.full((h, w), 255, dtype=np.uint8))
    alpha_mask = np.ascontiguousarray((mask_3d[:, :, 0] * 255).astype(np.uint8))

    tiled_f = generate_tiled_texture(
        pattern_tile_f,
        (h, w),
        best_scale,
        offset_x_percent,
        offset_y_percent,
    )
    tiled_bgr = (np.clip(tiled_f, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(tiled_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    pattern_alpha = np.ascontiguousarray(
        (np.clip((gray - alpha_thresh) * 10.0, 0, 1) * 255).astype(np.uint8)
    )

    adjusted_pattern_bgr = adjust_hsv_image(
        tiled_bgr,
        np.ones_like(mask_3d),
        hue_shift=best_h,
        sat_scale=best_s,
        val_scale=best_v,
    )
    highlight_overlay_map = create_highlight_overlay_map(lighting_map, mask_3d)

    orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    light_rgb = cv2.cvtColor((lighting_map * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    highlight_rgb = cv2.cvtColor(
        (highlight_overlay_map * 255).astype(np.uint8), cv2.COLOR_BGR2RGB
    )
    pattern_rgb = cv2.cvtColor(adjusted_pattern_bgr, cv2.COLOR_BGR2RGB)

    layer_bg = nested_layers.Image(
        name="Background",
        visible=True,
        top=0,
        left=0,
        bottom=h,
        right=w,
        channels={
            0: np.ascontiguousarray(orig_rgb[:, :, 0]),
            1: np.ascontiguousarray(orig_rgb[:, :, 1]),
            2: np.ascontiguousarray(orig_rgb[:, :, 2]),
            -1: alpha_full,
        },
    )
    layer_mask_img = nested_layers.Image(
        name="Mask",
        visible=True,
        top=0,
        left=0,
        bottom=h,
        right=w,
        channels={
            0: np.ascontiguousarray(orig_rgb[:, :, 0]),
            1: np.ascontiguousarray(orig_rgb[:, :, 1]),
            2: np.ascontiguousarray(orig_rgb[:, :, 2]),
            -1: alpha_mask,
        },
    )
    layer_pattern = nested_layers.Image(
        name="Pattern",
        visible=True,
        top=0,
        left=0,
        bottom=h,
        right=w,
        channels={
            0: np.ascontiguousarray(pattern_rgb[:, :, 0]),
            1: np.ascontiguousarray(pattern_rgb[:, :, 1]),
            2: np.ascontiguousarray(pattern_rgb[:, :, 2]),
            -1: pattern_alpha,
        },
    )
    layer_lighting = nested_layers.Image(
        name="Lighting",
        visible=True,
        blend_mode=BlendMode.multiply,
        top=0,
        left=0,
        bottom=h,
        right=w,
        channels={
            0: np.ascontiguousarray(light_rgb[:, :, 0]),
            1: np.ascontiguousarray(light_rgb[:, :, 1]),
            2: np.ascontiguousarray(light_rgb[:, :, 2]),
            -1: alpha_full,
        },
    )
    layer_highlight = nested_layers.Image(
        name="Highlight Overlay",
        visible=True,
        blend_mode=BlendMode.overlay,
        top=0,
        left=0,
        bottom=h,
        right=w,
        channels={
            0: np.ascontiguousarray(highlight_rgb[:, :, 0]),
            1: np.ascontiguousarray(highlight_rgb[:, :, 1]),
            2: np.ascontiguousarray(highlight_rgb[:, :, 2]),
            -1: alpha_mask,
        },
    )

    parsed_psd = nested_layers.nested_layers_to_psd(
        [layer_bg, layer_mask_img, layer_pattern, layer_lighting, layer_highlight],
        color_mode=3,
    )
    with open(output_path, "wb") as fd:
        parsed_psd.write(fd)


def save_result_bundle(
    output_dir,
    base_name,
    generated_bgr,
    orig_black_bgr,
    mask_3d,
    lighting_map,
    pattern_tile_f,
    alpha_thresh,
    scale_factor,
    offset_x_percent,
    offset_y_percent,
    h_shift,
    s_scale,
    v_scale,
):
    jpg_path = os.path.join(output_dir, base_name + ".jpg")
    psd_path = os.path.join(output_dir, base_name + ".psd")
    cv2.imwrite(jpg_path, generated_bgr)
    export_layered_psd(
        psd_path,
        orig_black_bgr,
        mask_3d,
        lighting_map,
        pattern_tile_f,
        alpha_thresh,
        scale_factor,
        offset_x_percent,
        offset_y_percent,
        h_shift,
        s_scale,
        v_scale,
    )
    return jpg_path, psd_path


def search_pattern_mockups(
    original_path,
    target_mask_path,
    pattern_tile_path,
    target_ref_path,
    output_dir,
    delta_e_threshold=7.0,
    export_hit_psd=True,
    base_scale=1.0,
    offset_x_percent=0,
    offset_y_percent=0,
):
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    orig_black_bgr = read_image_robust(original_path)
    if orig_black_bgr is None:
        raise FileNotFoundError(f"找不到底图: {original_path}")
    h, w = orig_black_bgr.shape[:2]
    mask_3d = preprocess_mask(target_mask_path, (h, w))

    pattern_tile_bgr = read_image_robust(pattern_tile_path)
    if pattern_tile_bgr is None:
        raise FileNotFoundError(f"找不到图案文件: {pattern_tile_path}")
    pattern_tile_f = pattern_tile_bgr.astype(np.float32) / 255.0

    dyn_gamma, dyn_clip, dyn_alpha, _ = get_dynamic_params(pattern_tile_bgr)
    lighting_map = extract_lighting_map(orig_black_bgr, mask_3d, dyn_gamma, dyn_clip)

    target_ref_bgr = read_image_robust(target_ref_path)
    if target_ref_bgr is None:
        raise FileNotFoundError(f"找不到参考图: {target_ref_path}")
    target_palette, target_weights = get_color_palette(target_ref_bgr, k=4, crop_center=True)

    best_coarse_diff = float("inf")
    best_coarse_params = (1.0, 0, 1.0, 1.0)
    coarse_scales = sorted(
        {
            max(0.1, round(base_scale - 0.10, 2)),
            max(0.1, round(base_scale - 0.05, 2)),
            max(0.1, round(base_scale, 2)),
            round(base_scale + 0.05, 2),
            round(base_scale + 0.10, 2),
        }
    )
    coarse_hues = range(-45, 46, 15)
    coarse_sats = [0.9, 1.1, 1.3]
    coarse_vals = [0.9, 1.0, 1.2, 1.4]

    for scale in coarse_scales:
        for h_shift in coarse_hues:
            for s_scale in coarse_sats:
                for v_scale in coarse_vals:
                    generated_bgr = apply_pattern_and_color_adjust(
                        orig_black_bgr,
                        mask_3d,
                        lighting_map,
                        pattern_tile_f,
                        dyn_alpha,
                        scale_factor=scale,
                        offset_x_percent=offset_x_percent,
                        offset_y_percent=offset_y_percent,
                        hue_shift=h_shift,
                        sat_scale=s_scale,
                        val_scale=v_scale,
                    )
                    gen_palette, _ = get_color_palette(generated_bgr, k=4, crop_center=True)
                    diff = calculate_palette_delta_e(target_palette, target_weights, gen_palette)
                    if diff < best_coarse_diff:
                        best_coarse_diff = diff
                        best_coarse_params = (scale, h_shift, s_scale, v_scale)

    c_sc, c_h, c_s, c_v = best_coarse_params
    fine_scales = sorted(
        {
            max(0.1, round(c_sc - 0.05, 2)),
            max(0.1, round(c_sc, 2)),
            round(c_sc + 0.05, 2),
        }
    )
    fine_hues = [c_h - 5, c_h, c_h + 5]
    fine_sats = [max(0.1, c_s - 0.1), c_s, c_s + 0.1]
    fine_vals = [max(0.1, c_v - 0.1), c_v, c_v + 0.1]

    best_fine_diff = float("inf")
    best_result = None
    hit_count = 0

    for sc in fine_scales:
        for h_shift in fine_hues:
            for s_scale in fine_sats:
                for v_scale in fine_vals:
                    generated_bgr = apply_pattern_and_color_adjust(
                        orig_black_bgr,
                        mask_3d,
                        lighting_map,
                        pattern_tile_f,
                        dyn_alpha,
                        scale_factor=sc,
                        offset_x_percent=offset_x_percent,
                        offset_y_percent=offset_y_percent,
                        hue_shift=h_shift,
                        sat_scale=s_scale,
                        val_scale=v_scale,
                    )
                    gen_palette, _ = get_color_palette(generated_bgr, k=4, crop_center=True)
                    diff = calculate_palette_delta_e(target_palette, target_weights, gen_palette)

                    current = {
                        "img": generated_bgr,
                        "scale": sc,
                        "h": h_shift,
                        "s": s_scale,
                        "v": v_scale,
                        "diff": diff,
                    }

                    if diff < best_fine_diff:
                        best_fine_diff = diff
                        best_result = current

                    if diff <= delta_e_threshold:
                        hit_count += 1
                        base_name = (
                            f"HIT_Sc{sc:.2f}_H{h_shift}_S{s_scale:.2f}_V{v_scale:.2f}_dE{diff:.2f}"
                        )
                        cv2.imwrite(os.path.join(output_dir, base_name + ".jpg"), generated_bgr)
                        if export_hit_psd:
                            export_layered_psd(
                                os.path.join(output_dir, base_name + ".psd"),
                                orig_black_bgr,
                                mask_3d,
                                lighting_map,
                                pattern_tile_f,
                                dyn_alpha,
                                sc,
                                offset_x_percent,
                                offset_y_percent,
                                h_shift,
                                s_scale,
                                v_scale,
                            )

    if best_result is None:
        raise RuntimeError("没有生成任何结果，请检查输入图片是否正常。")

    if hit_count == 0:
        base_name = (
            f"BEST_Sc{best_result['scale']:.2f}_H{best_result['h']}_"
            f"S{best_result['s']:.2f}_V{best_result['v']:.2f}_dE{best_result['diff']:.2f}"
        )
    else:
        base_name = (
            f"FINAL_BEST_Sc{best_result['scale']:.2f}_H{best_result['h']}_"
            f"S{best_result['s']:.2f}_V{best_result['v']:.2f}_dE{best_result['diff']:.2f}"
        )

    best_jpg_path, best_psd_path = save_result_bundle(
        output_dir,
        base_name,
        best_result["img"],
        orig_black_bgr,
        mask_3d,
        lighting_map,
        pattern_tile_f,
        dyn_alpha,
        best_result["scale"],
        offset_x_percent,
        offset_y_percent,
        best_result["h"],
        best_result["s"],
        best_result["v"],
    )

    return {
        "best_diff": best_result["diff"],
        "hit_count": hit_count,
        "best_scale": best_result["scale"],
        "best_jpg_path": best_jpg_path,
        "best_psd_path": best_psd_path,
        "best_preview_path": best_jpg_path,
        "elapsed_seconds": time.time() - start_time,
    }


def process_uploaded_files(
    original_file,
    mask_file,
    pattern_file,
    ref_file,
    input_dir,
    output_dir,
    delta_e_threshold=7.0,
    export_hit_psd=True,
    base_scale=1.0,
    offset_x_percent=0,
    offset_y_percent=0,
):
    original_path = save_uploaded_file(original_file, input_dir)
    mask_path = save_uploaded_file(mask_file, input_dir)
    pattern_path = save_uploaded_file(pattern_file, input_dir)
    ref_path = save_uploaded_file(ref_file, input_dir)

    return search_pattern_mockups(
        original_path=original_path,
        target_mask_path=mask_path,
        pattern_tile_path=pattern_path,
        target_ref_path=ref_path,
        output_dir=output_dir,
        delta_e_threshold=delta_e_threshold,
        export_hit_psd=export_hit_psd,
        base_scale=base_scale,
        offset_x_percent=offset_x_percent,
        offset_y_percent=offset_y_percent,
    )
