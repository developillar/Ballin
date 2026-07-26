/**
 * Flat-field test pattern.
 *
 * Several grade criteria — vignette strength, grain amplitude, chromatic
 * aberration — describe what the *post chain* does to the image. Measured on a
 * normal arena frame they cannot be separated from scene content: dark bowl
 * walls at the left and right edges read as 29% vignetting when the stack
 * applies none at all, and aliasing on high-contrast geometry reads as film
 * grain. Reviewers then go and fix effects that were never there.
 *
 * A flat field removes the ambiguity. The scene is replaced with a uniform
 * emissive surface filling the frame, the post chain runs over it unchanged,
 * and every deviation from flat in the captured image is something the stack
 * did. Vignette becomes an exact radial measurement, grain RMS is exactly the
 * grain, and R/B separation at the corners is exactly the aberration.
 *
 * Development only — nothing constructs this unless the harness asks for it via
 * `?scene=flatfield`.
 */

import {
  BackSide,
  Color,
  Mesh,
  MeshBasicMaterial,
  Object3D,
  SphereGeometry,
  type Scene,
} from 'three';

/**
 * Mid-grey, in the region the grain criterion is written about. Bright enough
 * that a vignette has something to darken and that grain sits in the mid-tones
 * where §8.5 specifies it, dark enough to stay well clear of clipping so the
 * tone curve is not compressing the very thing being measured.
 */
const FIELD_LUMINANCE = 0.42;

export class FlatField {
  private readonly hidden: Object3D[] = [];
  private shell: Mesh | null = null;
  private ui: HTMLElement | null = null;

  /**
   * Hides the scene and wraps the camera in a uniform shell.
   *
   * A shell rather than a full-screen quad because the post chain may consume
   * depth — for depth of field, or for any depth-aware pass — and a quad at the
   * near plane would hand it a degenerate depth buffer. A large sphere gives
   * every pixel a plausible depth while still being perfectly uniform.
   *
   * The HUD is hidden too when its root is passed: it is a DOM overlay, so it
   * lands in the screenshot and its panels would be measured as image content.
   */
  enable(scene: Scene, uiRoot?: HTMLElement): void {
    if (this.shell) return;

    if (uiRoot) {
      this.ui = uiRoot;
      uiRoot.style.visibility = 'hidden';
    }

    for (const child of scene.children) {
      // Lights stay: an unlit basic material ignores them, and removing them
      // would perturb anything in the stack that samples the light rig.
      if (child.visible && !(child as { isLight?: boolean }).isLight) {
        child.visible = false;
        this.hidden.push(child);
      }
    }

    const material = new MeshBasicMaterial({
      color: new Color(FIELD_LUMINANCE, FIELD_LUMINANCE, FIELD_LUMINANCE),
      side: BackSide,
      // No tone mapping, no fog: the point is to hand the post chain a known
      // constant, and anything applied before it is a variable in the result.
      toneMapped: false,
      fog: false,
    });
    this.shell = new Mesh(new SphereGeometry(60, 24, 16), material);
    this.shell.name = 'flatfield';
    this.shell.frustumCulled = false;
    scene.add(this.shell);
  }

  disable(scene: Scene): void {
    if (!this.shell) return;
    scene.remove(this.shell);
    this.shell.geometry.dispose();
    (this.shell.material as MeshBasicMaterial).dispose();
    this.shell = null;
    for (const child of this.hidden) child.visible = true;
    this.hidden.length = 0;
    if (this.ui) {
      this.ui.style.visibility = '';
      this.ui = null;
    }
  }
}
