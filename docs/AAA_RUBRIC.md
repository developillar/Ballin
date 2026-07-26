# AAA_RUBRIC — the Ballin visual standard

**Status:** normative. This is the yardstick. Every agent building a system in
this repo, and every critic reviewing a capture, is graded against this
document. If a criterion here conflicts with your taste, the criterion wins.
If a criterion here is wrong, change *this file* first and say so.

**The bar, stated once:** a knowledgeable observer — someone who watches NBA
basketball and plays sports games — is shown two still frames side by side, one
from NBA 2K (2K23–2K26 era, broadcast or 2K cam, in-engine gameplay, not
cutscene) and one from Ballin. They are told one is a shipped AAA title. They
cannot confidently pick which. That is a **9**. Anything less is a number below
9 and this document tells you which one and why.

---

## 0. How to use this document

### 0.1 The reference frame (RF)

All pixel figures in this document assume a **reference frame of 1080 × 2340
device pixels** (9:19.5 portrait — a 430 × 932 CSS px viewport at DPR 2.51,
rounded to a real device buffer). To apply a pixel criterion at another
resolution, scale by `H / 2340`.

Two canonical framings are referenced throughout. Both are portrait.

| Framing | Description | Approximate scale |
|---|---|---|
| **RIM** | Shooting/finishing view. Rim centre sits at ~34% of frame height from the top. Backboard occupies ~52% of frame width. Ball-handler is 1.5–4 m from camera. | ~530 px/m at the ball-handler |
| **FLOOR** | Half-court possession view. Ball-handler stands 6–11 m from camera, occupies ~24% of frame height. Near hardwood fills the bottom third. | ~290 px/m at the ball-handler, ~700 px/m at the near floor edge |

When a criterion says "at typical distance" it means the FLOOR framing.

### 0.2 Reading luminance

Criteria give post-grade **sRGB 8-bit display values (0–255)**. Sample with a
3×3 average, not a single pixel. "Clipped" means ≥ 252 in all three channels.

### 0.3 The three-pass review

A frame is reviewed in three passes, in this order. Do not skip ahead — most
failures are caught in pass 1 and arguing about pebble grain while the lighting
is flat is a waste of everyone's time.

1. **Squint pass.** Blur the frame to ~1/8 resolution mentally. Is there a
   readable light *shape*? Is the bowl dark and the floor bright? Is there a
   figure/ground separation between player and background? Most web-demo frames
   die here.
2. **Material pass.** At 100%, does every surface answer "what is this made of"
   with more than one cue (albedo + roughness variation + normal detail +
   specular shape)?
3. **Forensic pass.** The tells in §10. Hunt for them explicitly.

### 0.4 What is out of scope

Frame rate, input latency, animation quality *as gameplay*, audio. This rubric
is about what a **still frame** and a **short loop** look like. Motion is
covered only where it is legible from a 0.5 s clip (§9).

---

## 1. Lighting & exposure

Real reference: NBA arenas light the playing surface to roughly **1500–2000 lux
horizontal**, with broadcast specs calling for ~2000 lux average toward the main
camera and ~1700 lux average vertical illuminance on players. Colour rendering
is ≥ CRI 80 and correlated colour temperature sits around **5600 K** — neutral
daylight, not warm. Meanwhile the seating bowl is deliberately held dark: the
league-wide "theatre" lighting trend puts the crowd several stops under the
floor. That single ratio is the most important number in this document.

### 1.1 The bowl-to-floor ratio — the master criterion

- **Court mean luminance** (sample five points on lit hardwood inside the
  three-point line, away from painted areas): **95–140**.
- **Crowd mean luminance** (sample five points in the lower bowl, mid-frame,
  excluding LED boards and phone screens): **18–45**.
- **Ratio:** the court must be **2.5–4 stops** brighter than the bowl (6× to
  16× linear). Below 2 stops the frame reads as a lit room, not an arena. Above
  5 stops the crowd disappears into a black band and the frame reads as a
  practice-gym render with a void behind it.
- **Deepest arena shadow** (under the stands, behind the baseline): **6–16**.
  Never 0. A true-black region with no detail is a tell — real arenas have
  aisle lighting, exit signs and ambient bounce.
- **No large flat field.** Take any 200 × 200 px region of hardwood or crowd.
  Its standard deviation must exceed **6**. A region that samples to a single
  value is either untextured or unlit.

### 1.2 Light has direction, and there is more than one of it

- **Multiple shadows.** A player standing still on the court must cast **one
  dominant near-vertical contact shadow plus 2–5 fainter fanned shadows** from
  the surrounding overhead banks. The dominant shadow is at 25–45% of the
  adjacent floor luminance; the secondaries at 65–85%. **A single hard shadow
  is an instant fail** — it means one directional light.
- **Shadow softness scales with contact distance.** The penumbra at the sole of
  a planted foot is **≤ 3 px** (RF). The penumbra at the same player's raised
  hand, ~2 m off the floor, is **12–30 px**. Constant-softness shadows are a
  tell.
- **Vertical falloff.** Sample the same white jersey at chest height and at the
  shorts hem. The hem must be **10–25% darker** — overhead banks fall off, and
  the torso shadows the legs.
- **Key : fill on the player.** Measured on the same skin patch, lit side vs
  shadow side of the torso: **4:1 to 8:1**. Tighter than 3:1 is flat ambient.
  Wider than 12:1 is a stage, not an arena.
- **Rim light is mandatory and thin.** Along the top of the shoulders, the
  outer arm and the skull, a hot edge **2–6 px wide** (RF, FLOOR framing) at
  **1.4–2.5×** the luminance of the key-lit surface adjacent to it. It must
  break where the silhouette turns away from the back banks, not run uniformly
  around the whole outline. A uniform outline glow is a Fresnel hack and reads
  as such.

### 1.3 Colour temperature split

- **Overhead banks:** neutral to slightly cool. Target a white point around
  **5600–6000 K**; on a grey card at court level, B − R should be **+3 to +10**
  in 8-bit sRGB.
- **Hardwood bounce is the warm source.** Amber-warm, effective **3000–3800 K**.
  Checkable: sample the underside of a player's jaw, the shadow under the
  shorts hem, and the shadowed lower half of the ball. Each must be **warmer
  (R − B ≥ +8)** than the shadowed side of the same material higher up the
  body. If the bounce is neutral grey, the floor is not contributing and the
  frame will read as a studio turntable.
- **LED spill.** The courtside apron, the stanchion padding and the first two
  rows must carry a faint team-colour cast from the ribbon boards: **5–15%
  saturation**, not a coloured wash. It should be visible as a slight hue shift
  between the apron nearest the boards and the apron under the basket.
- **The bowl is cooler than the floor.** Crowd mean hue must sit **cooler** than
  court mean hue by a measurable amount (crowd B − R at least +6 relative to
  court B − R). Real crowds are lit by spill and phone screens, not by the
  banks.

### 1.4 Specular behaviour

- **Backboard glass:** two distinct specular populations. (a) A broad, soft
  reflection of the ceiling bank array — a blurred grid of 4–12 bright quads,
  luminance 60–120, occupying 20–50% of the glass area. (b) 1–3 **small hard
  speculars** at 200–255 from the nearest fixtures, each **6–20 px** across. If
  the glass shows only a uniform sheen it is a flat transparent plane and reads
  as plastic.
- **Chrome/steel** (stanchion arms, ring hardware, shot-clock frame): must show
  a **mirrored image of the environment with recognisable structure** — a dark
  bowl band, a bright ceiling band, a warm floor band. A chrome surface that is
  one grey value with a highlight dot is a tell.
- **Sweat specular on skin** is the sharpest specular in the frame after the
  glass. Individual highlights **2–8 px**, values 210–255, clustered on
  forehead, deltoids, upper back, forearms. They must sit *on top of* the
  subsurface diffuse, not replace it.
- **Nothing in frame has zero specular.** Not the crowd, not the padding, not
  the shorts. A matte-only surface is a missing shader.

### 1.5 Exposure discipline

- **Clipped pixels must be ≤ 1.5% of the frame** and must be confined to: LED
  boards, jumbotron, direct fixture reflections, sweat highlights, camera
  flashes. Clipped hardwood over more than 3% of the floor area means the
  exposure is blown.
- **Crushed pixels (≤ 4) must be ≤ 3% of the frame** and confined to deep
  under-stand geometry.
- **The histogram must be unimodal-with-a-tail**, not bimodal. A frame with a
  cluster at 20 and a cluster at 200 and nothing between reads as a poster.

---

## 2. Hardwood

Real reference: NBA floors are **hard maple** — tight-grained, light-coloured,
chosen partly because it reflects light and brightens the arena. A regulation
court is roughly **220 panels** of 3/4-inch maple; the milled strips are
narrow — **2 to 2-1/4 inches** face width is the norm — laid tongue-and-groove.
The full wood surface is about **120 × 60 ft** with the 94 × 50 ft court painted
on it. Lines and logos are painted onto sanded wood and then buried under
**multiple coats of high-gloss polyurethane**, refinished every season. This is
why an NBA floor looks wet on television.

### 2.1 Board layout and scale

- **Strip face width: 51–60 mm** (2 to 2-3/8 in). At the FLOOR framing near
  edge (~700 px/m) that is **36–42 px** per board; at mid-court (~290 px/m),
  **15–17 px**. A frame where you can count fewer than ~25 boards across the
  visible near floor has boards that are too wide — the single most common
  hardwood scale error.
- **Direction:** strips run **parallel to the sidelines** (baseline to baseline,
  i.e. along the court's long axis) in the default floor. A perpendicular or
  diagonal run must be a deliberate, stated design choice, not an accident of
  UV mapping.
- **Butt joints.** Strips are finite. Every **1.2–2.8 m** along a strip there
  must be an end joint, and joints in adjacent strips must be **staggered
  randomly**, never aligned into a visible column. Endless boards running the
  full 28 m are a tell.
- **Board-to-board tone variation.** Adjacent strips must differ in base
  lightness by **4–14 sRGB units** with an occasional (1 in ~12) strip
  differing by 20–30. Real maple floors have visible plank-to-plank colour
  scatter — 2K26 explicitly calls out reproducing "the distinct contrast in
  wood grain color." A floor where every board is the same tone is a tiled
  texture.
- **Seam line.** Between strips, a dark line **1–2 px** at near distance,
  fading to sub-pixel by mid-court. It must be a *tonal* seam plus a tiny
  specular break, not a black 3 px gap.
- **Panel grid.** Portable arena floors are assembled from panels (commonly
  ~4 × 7 ft). A very faint panel seam grid — a 1 px line at **25–40%** contrast
  of the board seams — should be discernible at grazing angles under strong
  reflection. This is a "how did they even" detail; it is not required below
  score 8 but it is the kind of thing that earns a 9.

### 2.2 Grain

- **Grain line spacing: 0.8–3.0 mm**, i.e. **15–25 grain lines across a single
  board's width** at close range. Hard maple is *tight* grained. Wide, wavy oak
  grain is wrong species and reads instantly as a stock texture.
- **Grain runs along the board's long axis** with gentle 2–6° wander and
  occasional cathedral figuring (1 board in ~8).
- **Grain must survive minification.** By mid-court the grain must not alias
  into moiré. Anisotropic filtering per `Quality.anisotropy` is mandatory on the
  floor; a shimmering floor under camera motion is a hard fail.
- **Grain is not just albedo.** The grain must appear in **roughness** as well —
  the dark late-wood lines sit fractionally rougher — so that when the floor
  catches a highlight the grain modulates the *highlight*, not only the base
  colour. If the grain vanishes inside a bright reflection, it is albedo-only.
- **Small knots and mineral streaks:** 1–4 visible per FLOOR frame, no more.
  Zero is sterile; a scattering of identical knots is a tiling artefact.

### 2.3 Clear coat and anisotropy

- **The floor is a two-lobe material:** a diffuse maple base plus a **smooth
  clear coat** on top. The coat's roughness is **0.06–0.14** in the specular
  streak direction and **0.18–0.30** across it — i.e. **anisotropic, stretched
  along the grain**. A perfectly isotropic circular highlight on hardwood is
  wrong.
- **Highlight shape:** reflections of the overhead bank quads must appear as
  **elongated streaks with a 2.5:1 to 6:1 aspect ratio**, long axis parallel to
  the board direction, soft-edged, with a brighter core.
- **Coat roughness varies spatially.** Traffic lanes (the paint, the top of the
  key, the wings) are **micro-scuffed**: locally rougher by 0.04–0.08, which
  shows as a slightly duller, wider highlight. The baseline corners and the
  area behind the basket stay glossier. A uniformly glossy floor is a tell.
- **The coat has a Fresnel edge.** At grazing angles (near the horizon line of
  the floor, bottom ~8% of a FLOOR frame) the reflection intensity must climb
  toward near-mirror — reflected player legs and rim hardware become clearly
  legible. Missing grazing-angle gain is one of the fastest ways to look like a
  web demo.

### 2.4 Reflections

- **Sharpness falls off with distance from the reflector.** A player's shoe
  reflects near-sharp within 20 px of the contact point; by 1.5 m up the
  reflected body is blurred to a soft vertical smear with **no readable
  features**. A crisp mirror image of the whole player is a raytraced-plane
  cheat and looks like a chess-set render.
- **Reflection intensity:** the reflected player at the contact point is
  **12–30%** of the direct player's luminance. Above ~40% it becomes an ice
  rink.
- **Reflections carry the bowl.** The floor must reflect the *dark* bowl and
  the *bright* ceiling, not just the player. Look at the hardwood in the top
  half of the FLOOR frame: it should carry a broad dark band from the far
  stands and a brighter band from the ceiling banks. A floor that reflects only
  the player is missing its environment.
- **Vertical stretch.** Reflections must be blurred **more vertically than
  horizontally** on an anisotropic coat — the streaks run with the grain.
- **No reflection may extend past its occluder.** The reflected foot must
  terminate exactly at the shoe. A reflection that leaks under a body is an
  SSR ray-march artefact and must be masked/faded.

### 2.5 Painted lines and logos — under the varnish

This is a signature test and it is easy to get wrong.

- **Paint sits under the coat.** Therefore: **a specular highlight crossing a
  boundary line must continue across it unbroken, at the same intensity and
  the same streak shape.** If the highlight dims, brightens, or changes shape
  when it crosses the sideline, the paint is on top and the frame is wrong.
- **Grain telegraphs through paint.** The wood grain must be faintly visible
  *through* the painted lines and the key — 8–20% of its contrast on bare wood.
  Perfectly opaque flat paint is a tell.
- **Line edges are crisp but not aliased:** hard-masked to within **1 px** of
  transition at near distance, but properly filtered. Line width is
  `COURT.lineWidth` (2 in = 51 mm) → **36 px** near, **15 px** mid-court.
  Lines that are one pixel wide at mid-court and shimmer are unfiltered.
- **Paint has its own micro-roughness.** Painted areas should be marginally
  *smoother* under the coat than bare wood (the paint fills the grain) — a
  1–4% brighter, slightly tighter highlight core over the key. Subtle, but its
  absence is what makes decals look pasted.
- **The centre logo** must (a) sit under the same coat and reflect the ceiling
  the same way, (b) show **grain telegraphing** through its lighter colours,
  (c) have edges with the same 1 px crispness as the lines, and (d) show
  **wear** — the logo lives at centre court and gets scuffed. A pristine,
  full-saturation, matte logo floating on a glossy floor is one of the top five
  tells in this document.
- **Logo saturation ceiling.** Painted court graphics are matte-mixed pigment
  under yellowing varnish. No painted region should exceed **~72% HSV
  saturation**; team colours read slightly muted and slightly warm compared to
  their brand values.

### 2.6 Wear, scuffs and life

- **Scuff distribution follows play, not noise.** Density is highest in the
  **paint / restricted area**, high on the **wings and top of the key**, low in
  the **corners**, near zero **behind the baseline**. A uniform noise overlay of
  scuffs across the whole floor is worse than no scuffs.
- **Scuff character:** short arcing black-rubber streaks, **60–350 mm** long,
  curved, at varied angles clustered around the direction of play. Contrast
  **6–20 sRGB units** below the local floor. Individually they are almost
  invisible; collectively they kill the plastic look.
- **Sole-print sheen patches.** Areas of very slightly *higher* roughness where
  shoes have polished/abraded the coat — visible only as highlight modulation,
  not as albedo.
- **Sweat spots.** 2–6 small (40–150 mm) darker, glossier wet patches per half
  court, plus the towel-wipe smears around them. Optional at score 7, expected
  at 9.
- **Dust in the low-traffic corners:** a 2–5% roughness increase near the
  boundary of the wood surface and against the apron.

---

## 3. Players

### 3.1 Silhouette and proportion

The silhouette is what a viewer reads first and it is the cheapest thing to get
wrong. `PLAYER.height = 1.98 m`, `PLAYER.shoulderWidth = 0.52 m`,
`PLAYER.reach = 2.6 m`.

- **Head count: 7.6–8.2 heads tall.** Generic humanoids are 7. An NBA player is
  long-limbed; a 7-head figure reads as a child at scale.
- **Shoulder : height = 0.25–0.28.** Waist width **0.40–0.48 of shoulder
  width**. Deltoid outer point is the widest part of the torso, not the ribcage.
- **Wingspan ≈ 1.02–1.08 × height.** Fingertip-to-fingertip must visibly exceed
  the standing height in an arms-out pose.
- **Leg length (crotch to floor) = 0.50–0.54 of height.** Short legs are the
  single most common proportion failure in procedurally generated figures.
- **Hand spans the ball.** Ball diameter is 0.2385 m; a spread hand must reach
  **0.20–0.23 m** across. If the hand looks small on the ball, the hand is
  wrong, not the ball.
- **Silhouette is never a smooth capsule.** Deltoid, lat, calf and quad must
  produce readable bumps on the outline. Trace the outline of the upper arm:
  it must be non-monotonic (bicep bulge, elbow narrowing, forearm swell).
- **Neck.** Trapezius must connect the neck to the shoulder as a slope, not a
  cylinder joining a sphere.
- **At least 3 distinct body types** should be discernible across a five-player
  frame — guard, wing, big. Identical bodies with different skins is a tell.

### 3.2 Muscle definition

- **Form shadows, not painted abs.** Definition must come from **normal detail
  + AO**, so it strengthens under a raking light and softens under flat light.
  Baked-into-albedo muscle shading is detectable because it doesn't change with
  light direction — check the same player in two lighting conditions.
- **AO in the anatomical creases:** armpit, under the pec, the linea alba,
  behind the knee, the Achilles. Ambient occlusion values 0.55–0.80 in those
  creases, never below ~0.4 (which reads as dirt).
- **Deformation on effort.** 2K26 explicitly ships flexed muscle definition
  after a dunk. Minimum bar for us: the **forearm and deltoid must visibly
  change shape** between a relaxed and a gripping/finishing pose. A body that is
  geometrically identical in every frame is a mannequin.
- **Tendon and vein detail** on the forearm and the back of the hand: present as
  fine normal detail at close range (RIM framing), gone by FLOOR framing.

### 3.3 Skin shading

The current 2K generation uses a dedicated skin shader that resolves sweat
streaming down individual pores and dry cracks in lips. We will not hit pore
level. We must hit the *light transport*, which is what actually reads.

- **Subsurface warmth at the terminator.** Where light wraps off the lit side
  into shadow there must be a **reddening band**: hue shifts toward 5–20°,
  saturation rises **8–20%** relative to both the lit side and the core shadow.
  Band width **6–20 px** at FLOOR framing on the shoulder/arm. This is the
  single most identifiable feature of real-time skin. Its absence makes skin
  read as painted plastic no matter how good the texture is.
- **Shadow terminator softness.** Skin's terminator must be **noticeably softer
  than the jersey's terminator on the same light**. If cloth and skin have the
  same falloff, there is no SSS.
- **Ear and finger translucency.** At RIM framing, backlit ears, nostril wings
  and the webbing between spread fingers must glow warm — value lifted 15–40
  above the surrounding shadow with strong red bias.
- **Two-lobe specular.** A broad, soft oil sheen (roughness ~0.4–0.55) over the
  whole skin surface, *plus* the sharp sweat highlights (roughness ~0.08–0.15)
  described in §1.4. One lobe only is a tell.
- **Sweat accumulates with time and exertion.** Specular intensity on the
  forehead/shoulders should be visibly higher late in a possession than at
  tip-off. Not required below 8; expected at 9.
- **Skin is not one colour.** Redder at the knuckles, elbows, knees, ears and
  nose; slightly desaturated and cooler at the palms and the soles-facing
  areas. Value variation across a single limb of at least **±6 sRGB units**
  before lighting.
- **Range of skin tones** across a squad must span meaningfully — and the SSS
  colour must shift with the tone (darker skin scatters less visibly and shows
  a stronger, tighter specular; do not just darken albedo and keep the same
  red terminator).

### 3.4 Jersey

- **Mesh weave is visible and it is small.** Hole pitch **1.2–2.2 mm** → at
  FLOOR framing (~290 px/m) that is **0.35–0.65 px** — i.e. **the weave must be
  a roughness/normal micro-detail that produces sheen break-up, not a visible
  grid**. If you can count holes at FLOOR framing, the weave is 3–5× too large.
  At RIM framing (~530 px/m) it is ~0.7–1.2 px and should read as fine texture.
- **Cloth specular is a broad, low, forward-scattering sheen** — roughness
  0.55–0.75 with a Fresnel-driven rim sheen on the shoulders and the outer edge
  of the fabric. Cotton/poly knit is **not** matte diffuse. A jersey with zero
  specular is the second-fastest way to look like a web demo after flat
  lighting.
- **Numbers and names are appliqué, not decals.** They must (a) sit **1–3 mm
  proud** of the base fabric with their own AO shadow along the lower/right
  edges, (b) carry **visible stitching** around the perimeter at RIM framing —
  2K26 ships "ultra-fine stitching" and curvature-accurate embroidered names,
  (c) have a **different roughness** from the mesh (tackle twill is smoother,
  glossier), and (d) **deform with the cloth** — a number that stays flat while
  the chest beneath it folds is a decal.
- **Drape and folds.** 3–7 readable folds across a torso at rest, concentrated
  at the waist where the jersey hangs off the shoulders and at the armhole. In
  a lateral lean, a **diagonal tension line from the leading shoulder to the
  trailing hip**. Folds must have both a shading gradient *and* a shifted
  specular, not just a darker line.
- **Hem behaviour.** The jersey hem must lift and lag in a drive, and settle
  after the stop. A rigid hem welded to the shorts is a tell.
- **Fabric thickness.** At the armhole, neckline and hem the cloth must show
  **finite thickness** — 1–3 px of edge at RIM framing — plus a visible bound
  trim. Zero-thickness cloth edges are paper.
- **Colour discipline.** Jersey colours under 5600 K arena light read slightly
  *less* saturated than their brand hex, and the white of a home jersey is
  **210–238**, not 255.

### 3.5 Shorts

- **Length:** hem lands **50–120 mm above the top of the patella**. Modern NBA
  cut. Too long reads 1990s; too short reads 1980s.
- **Fold behaviour:** shorts are heavier cloth than the jersey. They should show
  **fewer, larger folds** (2–4 per leg), a defined side-seam crease, and a
  visible waistband band of different material at 1.5–3× the thickness.
- **Motion:** the outer leg of the shorts swings with a **80–160 ms lag** behind
  the thigh and overshoots on a hard stop. Shorts rigidly parented to the leg
  bone are a tell and are visible even in a still, because the fabric will be
  perfectly conformal to the thigh.
- **Side panel / piping** must fold with the cloth.

### 3.6 Shoes

Shoes are close to the camera in almost every framing and they carry contact.

- **Silhouette:** midsole **28–38 mm** thick with a **visible bright outsole
  edge line** and its own contact shadow. A shoe that is one solid blob is a
  tell.
- **Three materials minimum:** knit/mesh upper (matte, fibrous), synthetic
  overlay or leather (glossier, roughness ~0.3), rubber outsole (roughness
  ~0.5, darker, with tread relief).
- **Laces** read as 2–5 px pale strokes at FLOOR framing with visible crossing.
- **Sock** must be visible between shoe collar and calf, with a ribbed cuff.
- **Outsole tread** must be visible on a raised foot (a jump, a follow-through)
  — herringbone or radial pattern, and it must catch the floor bounce.

### 3.7 Hair

- **Silhouette break-up is mandatory.** The outline of the hair must be
  irregular at a **2–8 px** scale. A smooth capsule/helmet is one of the top
  tells.
- **Rim light penetrates the outer 2–5 px** of hair, producing a halo that is
  brighter and warmer than the scalp.
- **Anisotropic sheen** along the hair flow direction on longer styles: a
  banded highlight, not a point highlight.
- **Facial hair** must break the jaw silhouette by 1–3 px and reduce specular
  on the skin under it.
- **At least 4 distinguishable hairstyles** across a squad; headbands, sleeves
  and arm bands as silhouette variety.

---

## 4. Ball

Real reference: the Wilson NBA game ball is an **eight-panel** full-grain
Horween leather ball with a **deep pebbled** cover produced by a molding
process, and **black channels (ribs) laid by hand to uniform depth and width**.
Circumference 29.5 in → `BALL.radius = 0.11925 m`, diameter 0.2385 m.

### 4.1 Scale on screen

| Framing | Ball diameter |
|---|---|
| RIM, in the shooter's hands (~530 px/m) | **126 px** |
| FLOOR, at the ball-handler (~290 px/m) | **69 px** |
| At the far basket (~120 px/m) | **29 px** |

Every pebble/seam criterion below must be evaluated against the relevant number.

### 4.2 Pebble grain

- **Pebble pitch 1.6–2.2 mm, bump diameter 1.0–1.5 mm.** At FLOOR framing that
  is **0.46–0.64 px** per pebble. Therefore: **the pebble must be a
  roughness-and-normal micro-detail that breaks up the sheen, not visible
  geometry.**
- **Correct appearance at FLOOR framing:** the specular band across the ball
  looks *grainy and irregular* — its edge is fuzzy and its intensity stipples —
  while the ball's albedo stays smooth. That is the whole effect.
- **Failure A — too big:** you can count individual pebbles at FLOOR framing.
  This is the golf-ball look and it is extremely common. If pebbles are
  individually resolvable at a 69 px ball, the texture is 3–5× oversized.
- **Failure B — absent:** the ball has a clean, smooth, circular specular
  highlight. This is a billiard ball.
- **Failure C — aliasing:** the pebble strobes or moirés as the ball spins or
  recedes. Mip and anisotropy must be correct; below ~35 px the pebble should
  have faded entirely into a roughness constant.
- At RIM framing (126 px) the pebble may become faintly resolvable as texture,
  never as bumps with their own cast shadows.

### 4.3 Seams and channels

- **Eight-panel layout.** Not six. Not a beach ball. The classic Wilson/Spalding
  pattern: two "capping" panels and a wrapping arrangement producing the
  familiar curved channel geometry.
- **Channel width 5–7 mm** → **1.5–2.0 px** at FLOOR framing, **2.7–3.7 px** at
  RIM.
- **Channels are recessed, not painted.** Each channel must show: a dark core
  (25–45% of the adjacent leather luminance), a **bright specular lip** on the
  side facing the key light (up to 1.6× adjacent), and an AO darkening on the
  opposite side. A uniform black line is a texture stripe.
- **Channel depth 1.5–3 mm** — enough that at RIM framing the channel
  visibly shadows.
- **The channel must not be pebbled.** Real channels are smooth rubber ribs.
  A ball whose pebble runs continuously through the seams is one texture with a
  stripe drawn on it.

### 4.4 Colour and sheen

- **Base leather:** a warm orange-brown, roughly hue 22–30°, saturation 62–78%,
  value 130–175 in the key. **Not** saturated safety orange. A traffic-cone
  ball is a tell.
- **Sheen falloff:** the specular band should be **broad and soft** (roughness
  0.30–0.45) with a **hard grazing-angle gain** at the silhouette edge — game
  leather is tacky and matte face-on, shiny at glancing angles. The bright rim
  around the ball's silhouette at 1.2–1.8× the face-on luminance is a signature
  of leather.
- **Wear:** the ball must be marginally shinier at the "equator" where hands
  and floor contact it most.
- **Floor bounce on the underside**, per §1.3, always.

### 4.5 In motion

- **Spin must be legible.** A shot ball carries **2.5–4 rev/s** of backspin. The
  channels must be visibly rotating; a ball that translates without rotating is
  a hard fail and is visible even in a still (the channel orientation should
  differ frame to frame).
- **Motion blur.** At 60 fps, a shot ball at 7–9 m/s covers 0.12–0.15 m/frame =
  **0.5–0.63 ball diameters**. With a 180° shutter, per-object motion blur must
  smear the ball by **~0.25–0.32 diameters** (17–22 px at FLOOR framing) along
  the velocity vector. A perfectly crisp ball at speed reads as a sprite.
  Motion blur is off below `high` per `Quality.motionBlur`; on `medium` and
  below, compensate with a slight roughness/contrast reduction rather than
  leaving a razor-sharp ball.
- **The ball keeps its contact shadow at all times.** A ball in flight 3 m up
  still casts a small, very soft (25–60 px penumbra), faint (85–92% of floor)
  shadow. A ball with no shadow floats; a ball with a hard shadow at height is
  equally wrong.

---

## 5. Hoop & net

### 5.1 Backboard glass

`HOOP.board`: 1.829 × 1.067 m, 38 mm thick, inner square 610 × 457 mm with a
51 mm border.

- **Transparency is not invisibility.** The glass must have: base transmission
  ~0.88–0.94, a **green-cyan tint that increases with path length**, and a
  **visible edge** where the glass thickness catches light.
- **Edge refraction.** At the 38 mm edges, a **2–5 px** band (FLOOR framing) of
  brighter, greener, distorted content. This is the highest-value single detail
  on the backboard and almost no web demo has it.
- **The crowd behind the glass must be visibly darkened and slightly shifted**
  relative to the crowd beside it. If the crowd reads identically through the
  glass and around it, the glass is a `transparent: true` plane.
- **Two specular populations** per §1.4.
- **The inner square is painted on the front face**, so it must (a) receive its
  own specular from the glass surface, (b) show a faint doubled ghost from the
  rear surface at oblique angles, and (c) have paint edges that are crisp within
  1 px.
- **Padding** under the glass: matte vinyl, roughness ~0.6, with visible seam
  stitching and a slightly compressed/creased profile. It must NOT be a
  perfectly clean extruded box.
- **The backboard is mounted, not floating.** The rear structure, the two
  mounting arms and the stanchion must be visible and must cast shadow on the
  glass and on the crowd behind.

### 5.2 Rim

`HOOP.rimRadius = 0.2286 m` (18 in inside diameter), `rimTubeRadius = 7.9 mm`
(5/8 in stock), `rimOffsetFromBoard = 152 mm`.

- **Tube diameter on screen:** 15.9 mm → **4.6 px** at FLOOR, **8.4 px** at RIM
  framing. It must be a **shaded tube with a highlight running along its top
  and a dark underside**, not a flat torus.
- **Colour:** NBA orange powder coat, hue 14–24°, saturation 70–85%, and
  **noticeably darker/duller than the ball** — sample both: the rim should be
  10–25 sRGB units darker in the same light.
- **Paint wear is mandatory.** The rim is the most abused painted object in the
  arena. Required: (a) **bare metal showing through on the top surface of the
  front third** of the ring where balls and hands hit — a 20–60% area of
  desaturated grey-silver with high specular, (b) **net-burn** darkening at each
  of the 12 net attachment points, (c) chipping along the ring's outer
  circumference. A pristine uniform orange ring is a top-five tell.
- **Hardware is visible:** the 12 net hooks/loops, the welded net-attachment
  ring or lugs, the breakaway hinge assembly behind the ring, and the bolt
  pattern where the rim meets the board. Missing hardware reads as a toy hoop.
- **The rim reads elliptically correct.** In portrait framings with the camera
  below rim height, the rim's screen ellipse minor/major ratio must match the
  camera pitch. Getting this wrong is a projection bug, not an art bug, but it
  is the first thing an experienced eye checks.

### 5.3 Net

`HOOP.net`: 0.381 m long, 12 strands, 9 segments, bottom radius 0.72× the rim.
Real nets are 120-count braided nylon with 12 stiffened anti-whip upper loops
and an hourglass profile.

- **Strand thickness on screen.** Real cord is ~3–4.5 mm → **0.9–1.3 px** at
  FLOOR framing, **1.6–2.4 px** at RIM. **This means the net must be rendered
  as sub-pixel-safe lines with correct alpha coverage.** Two failure modes:
  - **Too thick:** strands drawn as 3–5 px tubes. The net looks like rope or
    macramé. Extremely common.
  - **Aliased:** 1 px strands with no coverage AA, which crawl and sparkle
    under motion. Equally fatal.
  - Correct: strands that read as **soft, slightly translucent light-grey
    filaments** whose apparent brightness falls when they thin below a pixel,
    with the mesh reading as a **texture of light** rather than as geometry.
- **The net is not white.** Used nets are 200–235 at the top, dropping to
  180–210 at the hem, with grey-brown soiling at the bottom third and darkening
  at the top loops.
- **Diamond structure must be visible.** 12 strands × 9 rings gives a diamond
  mesh. From below (a common portrait framing) the crossings should be
  discernible as a lattice, not a haze.
- **Hourglass profile.** Wide at the rim, waisted at ~55–65% of the length,
  flaring slightly at the hem. `bottomRadiusScale = 0.72` is the hem; the waist
  must be tighter than both ends. A straight cone is a hard fail and is
  explicitly one of the named tells.
- **The net hangs, it does not stand.** At rest, gravity sag must be visible:
  the strands are near-vertical at the top and the hem sits ~2–5 mm inside where
  a straight taper would place it.
- **Backlighting.** With the ceiling banks above, the net must be **brighter
  than the crowd behind it** and should slightly bloom. A net that is darker
  than its background is unlit.

### 5.4 Net motion on a make

The 2K-family approach is a particle system with Verlet integration, fixed and
stiff-spring constraints, continuous collision against the ball, custom
collision reaction and a post-integration correction pass specifically to stop
the ball tunnelling through the net. Match the *behaviour*, at
`Quality.netIterations` per tier.

- **Ball entry:** the net must **balloon outward** ahead of the ball — the
  widest point travels down the net *ahead of* the ball's centre.
- **Hem flip:** on a clean swish the hem must **flip upward, briefly rising
  above the rim plane or at least to within 60 mm of it**, then fall back. If
  the net never inverts, it is not simulated.
- **Settle:** **2–3 decaying oscillations over 0.35–0.6 s**, each ~45–60% of the
  previous amplitude, ending fully at rest. A net that stops in one frame is
  keyframed; a net that oscillates for 2 s is under-damped.
- **Asymmetry.** An off-centre make must produce an asymmetric response — the
  side the ball touched moves more.
- **Rim contact:** a ball that hits the ring must **flex the rim visibly** —
  2–8 mm of tip-down deflection on the near side, recovering over ~0.15–0.25 s
  with one small overshoot — and the net must swing from the impulse.
- **The net never intersects the ball.** Not for a single frame. Interpenetration
  is instantly visible and instantly disqualifying.

---

## 6. Arena & crowd

2K's stated goal is that all 30 arenas closely match their real counterparts.
We have one fictional arena; the standard is not accuracy but **plausibility as
a photographed room**.

### 6.1 The bowl

- **Darkness per §1.1** — 2.5–4 stops under the court. This is the criterion.
- **The bowl must have architecture**, legible even at 20–45 luminance: a
  visible seating rake, aisle steps, vomitory openings (darker rectangles),
  a lower-bowl/upper-bowl break, a suite band, railings. A flat dark gradient
  is a backdrop, not a building.
- **Depth cueing.** Far stands are **lower contrast and slightly hazier** than
  near stands — a subtle atmospheric term (0.02–0.06 density over 30 m) sells
  the volume of the room. Zero haze makes a large space look like a small one.
- **The ceiling exists.** Overhead, the frame must be able to show truss work,
  the light banks themselves as bright quads, catwalks and hanging speaker
  arrays. A black void above the rim in a RIM framing is a missed opportunity
  and is noticeable when the camera tilts.
- **The apron.** The 2.9 m × 2.3 m painted/floor apron beyond the lines
  (`COURT.apronX/apronZ`) must be present, must be the **same varnished
  hardwood** as the court (usually stained darker or carrying sponsor
  graphics), and must carry the LED colour spill. The court must not end at the
  boundary line — that is a hard fail.

### 6.2 LED ribbon and jumbotron

- **The ribbon boards are the brightest continuous elements in the frame** after
  direct fixtures: **190–250**, saturated team colour, with visible horizontal
  banding/scan structure at close range.
- **They must bloom** (see §8.1) and must **cast coloured light** onto the apron
  and the first rows (§1.3). A bright board that lights nothing is an emissive
  quad, and it looks like one.
- **They must show content**, even abstract: moving colour blocks, a score
  strip, sponsor bands. A single flat colour bar is a placeholder.
- **The jumbotron** is a large bright rectangle high in frame. It must be
  **out of the depth-of-field plane** in a FLOOR framing (soft), must bloom, and
  must be dimmer per-pixel than the ribbon boards' peak but larger in area.
- **Aspect and pixel pitch:** any LED surface at close range must show a
  **pixel grid** — a dark lattice at 5–15% contrast. Smooth gradients on an LED
  wall are wrong.

### 6.3 Crowd

`Quality.crowdCount` ranges 260 (low) → 2600 (ultra), `crowdAnimated` off only
at low. Budget is not an excuse for the failures below.

- **Density reads as continuous.** At FLOOR framing there must be **no visible
  gaps in a regular pattern** and **no visible grid**. If you can see rows and
  columns of evenly spaced figures, it is a billboard array — a named tell.
- **Silhouette variety.** Across any 100 visible crowd members: varied head
  heights (±120 mm), varied shoulder widths, varied poses (standing, seated,
  leaning, arms up, on a phone), and **at least 8 distinguishable colour
  clusters** in clothing. Identical silhouettes repeated in a lattice is fatal.
- **Occlusion between crowd members.** Nearer spectators must overlap and
  occlude farther ones. Non-overlapping figures are a spaced grid.
- **Seats are visible.** Empty seats (there are always some), seat backs, the
  colour of the seating, and the gaps between rows. A crowd with no visible
  seating furniture is a texture.
- **Motion, not synchrony.** With `crowdAnimated`, motion must be
  **desynchronised** — phase offsets spread across the full cycle, varied
  amplitude, varied frequency (±25%). A crowd bobbing in unison is worse than a
  static crowd.
- **Do not resolve faces.** At FLOOR framing a crowd member is 10–25 px tall;
  faces must be **skin-coloured blobs with a value gradient**, not features. A
  crowd with legible eyes at that size is uncannily wrong.
- **Phone screens.** 5–20 small (2–4 px) bright cool-white/blue points scattered
  in the bowl, each with a faint bloom. Cheap, and it sells the darkness
  instantly.
- **The crowd must not be brighter than the near hardwood.** Ever.

### 6.4 Courtside

The first 2–4 rows are the closest, most-lit and most-scrutinised part of the
non-playing environment.

- **Individually posed 3D figures**, not impostors, for the front two rows in a
  FLOOR framing. They get the same rim light and floor bounce as the players.
- **Furniture:** courtside chairs/seats, the scorer's table with its own LED
  face, camera operators with visible tripods/lenses, ball boys with towels,
  the bench with a row of seated players and coaches, water bottles, towels,
  a stanchion of photographers along the baseline with occasional **camera
  flashes** (1–3 per second across the frame, 2–6 px, clipped white, with
  bloom).
- **The baseline sponsor board** and the padded stanchion base.
- **Everything courtside sits on the hardwood apron and reflects in it.**

---

## 7. Camera & composition

Portrait 9:19.5 is not a cropped landscape frame. The composition rules are
different and this section is where most portrait sports demos fail.

### 7.1 Framing targets

- **RIM framing.** Rim centre at **30–38% of frame height from the top**;
  backboard occupying **45–60% of frame width**. Ball-handler's head between
  **58% and 72%** of frame height. The bottom **12–18%** is reserved for near
  hardwood and HUD safe area.
- **FLOOR framing.** Ball-handler's feet at **62–74%** of frame height. The
  target basket must be visible in the **upper 35%** of the frame. Near
  hardwood occupies the bottom **20–30%**.
- **The vertical axis carries the action.** In portrait, depth maps to screen-Y.
  The frame must be composed so that the ball's flight is a **vertical or
  near-vertical arc** through the tall dimension. A shot arc that exits the side
  of the frame is a camera failure.
- **HUD safe areas:** top 8% and bottom 22% of the frame may be overlaid.
  Nothing critical (rim, ball apex, shooter's hands) may live there.
- **Headroom discipline.** Never fewer than 4% of frame height above the tallest
  in-frame head, never more than 22% of dead space above it.

### 7.2 Focal length feel

- **Target a 35–50 mm full-frame-equivalent horizontal FOV**, i.e. a **vertical
  FOV of roughly 42–58°** in 9:19.5. Three.js `PerspectiveCamera.fov` is
  vertical: **45–55** for FLOOR, **38–46** for RIM (slightly longer, flatter,
  more broadcast).
- **Too wide (fov > 65) is the classic web-demo look**: exaggerated perspective,
  banana-shaped court lines, a player's near foot twice the size of the far
  foot, and a floor that appears to tilt away steeply. Check by comparing the
  screen height of the near and far sidelines — the ratio must be **≤ 3.2:1**.
- **Compression cue:** at correct focal length, a defender 2 m behind the
  ball-handler should be **~80–90%** of the handler's screen height, not 55%.
- **Camera height 1.1–2.2 m** for FLOOR (roughly a courtside/low-seat eyeline),
  **1.4–2.6 m** for RIM. A camera at 6 m looking down is an RTS view and reads
  as one instantly.
- **Slight downward pitch, never level-zero.** 3–12° down for FLOOR.

### 7.3 Inertia and lead

Legible from a short clip, not a still, but decisive.

- **The camera never snaps.** All camera position and orientation targets are
  smoothed with a critically-damped spring, settle time **180–400 ms**. A
  camera that is rigidly parented to the ball is the most nauseating and most
  obviously amateur thing in a sports demo.
- **The camera leads.** Target point is offset **0.15–0.45 s** ahead along the
  ball/handler velocity — the action sits slightly behind centre in the
  direction of travel.
- **Overshoot is allowed, once.** A single ≤ 6% overshoot on a hard direction
  change reads as a real operator. Repeated oscillation does not.
- **Micro-handheld.** A **very** small procedural noise on rotation (amplitude
  ≤ 0.12°, 0.4–1.2 Hz, two octaves) makes the frame feel operated. Above 0.3°
  it becomes seasickness. This is optional but it is a 9-level detail.
- **Cut discipline:** any cut must be > 1.2 s from the previous cut. Rapid cuts
  hide nothing and read as a bug.

### 7.4 Depth of field

Broadcasters have moved hard toward shallow depth of field for NBA coverage —
cine bodies with 50 mm and 24–70 mm glass alongside the long box lenses. Use it,
but sparingly; `Quality.depthOfField` is off below `high`.

- **Focus plane on the ball-handler**, tracked with a 120–250 ms lag (a real
  focus puller is never instant).
- **Circle of confusion at the crowd: 8–22 px** at FLOOR framing. Enough that
  faces dissolve, not so much that the bowl becomes an abstract wash.
- **Near hardwood at the very bottom of the frame: 3–8 px** CoC. A little near
  blur anchors the viewer's eye position.
- **The rim and the shooter must both be acceptably sharp in a RIM framing** —
  do not blur the thing the player is aiming at.
- **Bokeh must be round-ish and must bloom on point highlights** (camera
  flashes, phone screens). Gaussian mush with no highlight structure is a
  give-away.
- **DOF must not bleed across silhouette edges.** A halo of background colour
  wrapping a sharp player is a naive single-pass DOF and is very visible.

---

## 8. Post & grade

An ungraded render looks like a render. This section is cheap in performance and
enormous in perceived quality, which makes failures here inexcusable.

### 8.1 Bloom

`Quality.bloom` is on at every tier; `bloomMips` 3 → 6.

- **Threshold in scene-referred luminance, not display.** Bloom should engage
  only above roughly **1.15–1.5×** display white in the linear buffer. If
  ordinary lit hardwood glows, the threshold is too low.
- **Bloom sources should be countable:** LED boards, jumbotron, direct fixture
  reflections in the glass and chrome, sweat/skin peak speculars, camera
  flashes, the brightest varnish streaks. Everything else must be clean.
- **Shape: wide and faint, not tight and bright.** Multi-mip: the largest mip
  should reach **12–25% of frame height** at **2–6% intensity**; the tightest
  mip **6–20 px** at **20–40%**. A single-radius gaussian halo is the
  characteristic "WebGL bloom" and it is a named tell.
- **Total bloom energy budget:** if you difference the pre- and post-bloom
  frames, the mean lift must be **≤ 4 sRGB units**. Anything more is a haze
  filter and it destroys contrast.
- **Bloom must not wash the bowl.** Measure crowd mean before and after bloom;
  the lift must be ≤ 3 units, or the 2.5-stop ratio in §1.1 is being eaten.

### 8.2 Tone curve

- **Use a filmic/ACES-like curve**, not Reinhard, and never a straight
  clamp. Required characteristics: a **toe** that lifts the deepest blacks
  slightly off zero, a roughly linear mid, and a **long shoulder** that
  compresses 1.0–8.0 scene-linear into the top 15% of display range.
- **Highlight desaturation.** As values approach white, saturation must
  **fall** — the core of a clipped LED board should be near-white, ringed by
  its saturated colour. Colours that stay fully saturated at 255 are the
  signature of a missing tonemapper.
- **Black point:** deepest frame value 5–14, not 0.
- **Skin must sit on the linear part of the curve.** Mid-tone skin in the key at
  120–190 (§1.1). Skin on the shoulder means blown faces.

### 8.3 Shadow / highlight colour split

- **Shadows cool:** lift the blue channel in the bottom 25% of the range —
  target hue 195–225°, **4–10% saturation**. Measure: in the darkest quartile of
  the frame, mean B − R should be **+5 to +14**.
- **Highlights warm:** hue 30–48°, **3–8% saturation** in the top 20% of the
  range. Mean R − B **+4 to +12**.
- **The split must be subtle.** If a reviewer can name the colour of the shadows
  without measuring, it is 2–3× too strong and reads as an Instagram filter.
- **Mid-tone hue must stay honest.** The hardwood must still read as maple and
  the ball must still read as leather.

### 8.4 Vignette

- **Strength:** corners **10–22% darker** than centre. Below 8% it does nothing;
  above 28% it reads as a cheap overlay.
- **Shape:** elliptical matching the 9:19.5 aspect, with a **soft, wide falloff
  starting at ~55% of the frame radius**. A tight circular vignette on a very
  tall frame produces dark bands top and bottom and looks like a phone-app
  filter.
- **Vignette must be applied before grain, after bloom.**
- **Slight desaturation** in the vignette region (3–8%) is authentic to real
  optics and is a nice-to-have.

### 8.5 Grain

`Quality.filmGrain` is on at every tier, including `low`.

- **Amplitude: 1.5–4 sRGB units** RMS in mid-tones. Enough to break gradient
  banding, not enough to be described as "noisy."
- **Luminance-weighted:** strongest in the mid-tones and shadows, **suppressed
  in the top 10%** of the range. Grain crawling over a blown LED board is
  wrong.
- **Grain must be applied in screen space at output resolution** and must
  **not** be scaled by the adaptive render scale — grain that grows chunky when
  `AdaptiveGovernor` drops resolution is an immediate tell.
- **Grain should be slightly chromatic** (per-channel offset), not pure
  monochrome noise.
- **Grain animates every frame.** Static grain is a dirty lens.

### 8.6 Chromatic aberration

`Quality.chromaticAberration` on at `medium` and above.

- **Radial, zero at the centre**, growing to a maximum of **0.8–2.0 px** of R/B
  separation at the frame corners.
- Above ~3 px it is a music video. Below ~0.5 px it does nothing; that is fine —
  prefer too little to too much.
- **Must be applied before grain** and must **not** apply to HUD elements.

### 8.7 What must NOT be in the stack

- Global contrast/saturation boosts that push the crowd up out of §1.1.
- A "sharpen" pass with visible halos on silhouette edges.
- Lens flares with anamorphic streaks. Not a 2K look.
- A full-frame colour overlay (blue/orange "cinematic" wash).
- Screen-space god rays from the ceiling. `volumetricLight` is for subtle
  atmosphere in the upper bowl only; visible light shafts over the court are
  a concert, not a basketball game.

---

## 9. Motion & contact

Assessed from a 0.5–1.5 s loop. Several of these are visible in a still.

### 9.1 Foot plant and ground contact

- **The contact shadow meets the shoe.** Zero gap. A 1 px gap between sole and
  shadow is visible and is the single most common "floating" failure. Check
  every planted foot in the frame.
- **Contact darkening.** Immediately under the sole (within 30 mm) the floor
  must be **35–55%** of its unoccluded luminance, rising to 70–85% at 150 mm.
  This is contact AO and it is what makes weight read.
- **The reflection meets the shoe too** (§2.4).
- **Planted feet do not slide.** Over a 0.5 s clip, a foot in contact must have
  **≤ 15 mm** of world-space drift. Skating is fatal and unmistakable.
- **The sole conforms.** A foot on the floor must not intersect it and must not
  hover — the visible sole plane must be coplanar with the floor within 2 mm.
- **Weight shift.** In a stance, the supporting hip must be **higher** than the
  free hip and the spine must show a compensating curve. A symmetric,
  perfectly-vertical standing pose is a T-pose with the arms down.

### 9.2 Anticipation and follow-through

- **Every impulsive action has a wind-up.** A jump shot shows a dip (60–140 ms)
  before extension. A pass shows a load. A cut shows a plant and lean. Motion
  that starts at full speed from rest is linear interpolation and reads as such.
- **Follow-through.** The shooting wrist must snap and **hold** — the hand stays
  in the follow-through pose for 250–500 ms. The off-hand releases first.
- **Overshoot on stops.** Deceleration (`PLAYER.deceleration = 30 m/s²`) must
  produce a visible torso lean opposite to travel, then a settle.
- **Secondary motion everywhere.** Jersey hem, shorts, hair, the head
  counter-rotating slightly against the shoulders. A body with zero secondary
  motion is rigid even when the primary animation is perfect.
- **No pops.** No joint may change orientation by more than ~35° in a single
  frame at 60 fps, outside a deliberate cut.

### 9.3 Ball handling and contact

- **The hand touches the ball.** Fingers must contact the surface with visible
  compression of neither (no interpenetration, no gap). At RIM framing a gap of
  3 px between fingertip and leather is glaring.
- **Dribble contact.** On a dribble the ball must reach the hand, not stop 40 mm
  short. The hand must ride the ball up and push it down.
- **Ball shadow tracks the ball** at all times, with penumbra scaling with
  height (§4.5).

### 9.4 Rim, net and board reaction

- Per §5.4. Restating the disqualifiers: a net that does not invert on a swish;
  a rim that does not flex on contact; a ball that passes through net geometry;
  a net that stops instantly.
- **The backboard reacts to a hard hit** — a small (2–5 mm) shake with a fast
  (0.2–0.3 s) decay, transmitted to the net and the padding.
- **The stanchion never moves.** It is bolted and weighted.

---

## 10. The tells

This is the list a hostile reviewer reads first. Each of these, on its own,
identifies a frame as a hobby WebGL demo rather than a shipped AAA sports
title. Finding any one of them caps the relevant category at **4**.

### Lighting tells

1. **Uniform ambient light with no directional shape.** The frame is evenly lit,
   nothing has a lit side and a shadow side, and the whole image reads flat.
   This is number one for a reason.
2. **A single hard shadow per player** — one directional light in an arena with
   dozens of fixtures.
3. **No rim light.** Players' silhouettes dissolve into the background because
   nothing separates them from it.
4. **A rim light that is a uniform Fresnel glow** all the way around the
   silhouette regardless of where the lights are.
5. **The crowd is as bright as the court.** The 2.5-stop ratio is missing and
   the arena reads as a gymnasium at noon.
6. **A pure black void** where the bowl or ceiling should be.
7. **No floor bounce.** Undersides of chins, shorts hems and balls are the same
   neutral grey as every other shadow.
8. **Everything is the same colour temperature.** No cool/warm structure at all.

### Material tells

9. **Flat untextured skin** — a single albedo colour with a Lambert term and
   maybe one specular dot. No SSS, no terminator warmth, no sweat.
10. **A perfectly clean floor.** No scuffs, no wear pattern, no sweat, no dust,
    no board-to-board tone variation. It looks like laminate flooring from a
    material library.
11. **Isotropic circular highlights on the hardwood** instead of grain-aligned
    streaks.
12. **Hardwood boards that are 3–5× too wide** — twelve fat planks across the
    court instead of a hundred narrow strips.
13. **Painted lines that sit on top of the varnish** — the specular streak dims
    or vanishes when it crosses a line.
14. **A pristine, fully saturated centre logo** with matte finish, sitting on a
    glossy floor like a sticker.
15. **A jersey with zero specular** — pure diffuse cloth. No sheen, no weave, no
    Fresnel edge.
16. **Numbers as flat decals** that do not deform with the fabric and have no
    stitching or relief.
17. **A basketball that is a smooth orange sphere** with a black stripe texture.
    No pebble, no recessed channel, no grazing-angle sheen.
18. **Golf-ball pebble** — pebble grain 4× oversized and individually
    resolvable at gameplay distance.
19. **Chrome that is a grey value with a highlight dot** instead of a mirrored
    environment.
20. **Backboard glass as a `transparent: true` plane** — no edge thickness, no
    refraction, no tint accumulation, crowd unaffected behind it.
21. **A pristine orange rim** with no bare-metal wear and no visible hardware.

### Geometry / silhouette tells

22. **A net that is a static cone.** No hourglass waist, no sag, no simulation.
23. **A net drawn as thick rope** — 4 px strands where 1 px is correct.
24. **Players who are 7 heads tall with short legs** and capsule limbs with no
    muscle break-up in the silhouette.
25. **Helmet hair** — a smooth, closed, un-broken hair silhouette.
26. **Shoes as single blobs** with no midsole line, no laces, no sock.
27. **Zero-thickness cloth** at jersey armholes and hems.
28. **The court ends at the boundary line** with no apron — the wood surface is
    exactly 94 × 50 ft and then stops.
29. **A hoop with no stanchion**, floating backboard, or missing net hooks.

### Crowd / arena tells

30. **A crowd that is obviously flat billboards in a grid** — even spacing,
    identical silhouettes, no occlusion between figures, visible rows and
    columns.
31. **A crowd that bobs in perfect unison.**
32. **Crowd members with resolvable faces** at 15 px tall.
33. **No seats** — figures floating on a dark rake.
34. **LED boards that are flat emissive quads** with no content, no pixel grid,
    no bloom and no light contribution to the scene.
35. **No courtside layer at all** — the players are on a floor in a dark room.

### Camera / post tells

36. **A 75° FOV** producing exaggerated perspective, curved-looking lines and a
    steeply tilting floor.
37. **A camera rigidly parented to the ball** with no smoothing, no lead and no
    inertia.
38. **A camera 6 m up looking down** — RTS view, not broadcast.
39. **Landscape composition letterboxed or cropped into portrait**, with the
    action stuck in a narrow horizontal band in the middle of a tall frame.
40. **Single-radius gaussian bloom** producing a uniform glowing haze over
    everything, including the crowd.
41. **No tonemapper** — a raw linear buffer clamped to 1.0, with saturated
    colours surviving all the way to white.
42. **No grain**, so every gradient bands visibly.
43. **Grain that scales with the adaptive render scale** and gets chunky under
    load.
44. **A heavy blue or orange full-frame wash** standing in for a grade.
45. **Lens flares.**

### Contact / motion tells

46. **Feet that float** — a visible gap between sole and shadow.
47. **Foot skating** during any locomotion.
48. **A ball with no shadow**, or with a hard shadow while 3 m in the air.
49. **A ball that translates without spinning.**
50. **The ball passing through the net geometry**, even for one frame.
51. **A rim that does not move when the ball hits it.**
52. **Poses that snap** with no anticipation and no follow-through.
53. **Zero secondary motion** — jersey, shorts and hair rigidly welded to bones.

---

## SCORING SHEET

Ten categories, each **0–10**. Score the frame, not the intention.

**Scale anchors, universal:**

| Score | Meaning |
|---|---|
| 0–2 | Placeholder. Untextured, unlit, or absent. |
| 3 | Recognisable but obviously a hobby WebGL demo. Contains multiple §10 tells. |
| 4–5 | A competent web demo. One or two §10 tells remain. Nobody would mistake it for a shipped title. |
| 6 | A good web game — the best of its class on the open web. Still identifiable as a browser render within ~3 seconds. |
| 7–8 | Reads as a shipped console game, but a generation or two behind 2K, or with one category conspicuously lagging. Identifiable in ~10 seconds by someone looking for it. |
| 9 | **The bar.** Blind A/B against a real NBA 2K gameplay frame: a knowledgeable observer cannot confidently pick which is which. |
| 10 | Beats 2K on this axis. Reserved. Do not award it to your own work. |

**Overall gate:** the project score is **the minimum of the ten category
scores**, not the mean. One flat-lit frame with a perfect floor is a flat-lit
frame. Report the mean as a secondary figure only.

---

### 1. Lighting & exposure

- **3** — Ambient + one directional. One hard shadow per player. Crowd and court
  within a stop of each other. No rim light, no colour temperature structure,
  no bounce. The scene is evenly visible and completely shapeless.
- **6** — A real key with soft shadows, a fill, and a rim light. The bowl is
  visibly darker than the court, maybe 1.5–2 stops. Colour temperature split
  exists but is either absent in shadow or overcooked. Players cast one shadow.
  Bounce light is a flat ambient term rather than directional warmth from the
  floor. Reads as "well lit 3D," not as "photographed arena."
- **9** — The full §1 spec. 2.5–4 stop bowl-to-court ratio. Multi-bank shadows:
  a dominant contact shadow plus fainter fans. Penumbra scales with contact
  distance. Rim light is thin, hot, and breaks correctly around the silhouette.
  Neutral 5600 K key against warm hardwood bounce that measurably warms every
  downward-facing surface. LED colour spill on the apron. Two specular
  populations on glass; chrome carries a readable environment. Sweat specular on
  top of subsurface diffuse. Clipping ≤ 1.5% and confined to legitimate sources.
  A photographer would look at the frame and read the lighting plot.

### 2. Hardwood

- **3** — A brown plane with a wood texture, one roughness value, an isotropic
  highlight, and lines painted on top as decals. Boards, if visible, are far too
  wide. Perfectly clean. Possibly a visible texture tile repeat.
- **6** — Correct board width and direction, plausible maple grain, a glossy
  clear coat with some reflection, and lines that read as painted rather than
  stuck on. But: isotropic highlight, uniform gloss everywhere, no butt joints,
  no board-to-board tone variation, no wear pattern, and a clean centre logo.
  Reflections are either absent or a uniform mirror.
- **9** — 51–60 mm strips with staggered butt joints and per-board tone scatter.
  Tight maple grain in albedo *and* roughness, filtered clean at all distances.
  Anisotropic clear coat producing grain-aligned streak highlights with a
  correct grazing-angle Fresnel gain. Distance-dependent reflection blur that
  carries the dark bowl and the bright ceiling, not just the players.
  Play-weighted scuffs, traffic-lane roughness variation, sweat spots. Lines and
  logo under the varnish, with grain telegraphing through and highlights
  crossing them unbroken. The logo is worn. The floor looks like it has hosted
  40 games this season.

### 3. Players

- **3** — Capsule-limbed 7-head humanoids with flat skin, flat cloth, blob
  shoes, helmet hair, and geometry that is identical between poses. Silhouette
  reads as a mannequin.
- **6** — Correct NBA proportions, readable muscle in the silhouette, cloth
  with visible folds and some sheen, shoes with a midsole. But skin is a
  diffuse+specular material with no subsurface warmth at the terminator; jersey
  numbers are decals; hair silhouette is closed; shorts are conformal to the
  thigh; everyone has the same body type.
- **9** — 7.6–8.2 heads, correct wingspan and leg ratio, three-plus body types.
  Muscle from normals+AO that responds to light direction and changes shape on
  effort. Skin with a measurable warm terminator band, softer falloff than
  cloth, translucent ears and finger webbing, two-lobe specular with sweat
  clusters that build over a possession. Jersey with sub-pixel weave breaking
  up a Fresnel-edged sheen, appliqué numbers with stitching and relief that
  deform with the cloth, finite fabric thickness at every edge, correct drape
  and a lagging hem. Shorts with heavier fold behaviour and visible lag. Shoes
  with three materials, laces, socks and tread. Hair with an irregular
  silhouette and anisotropic banded sheen.

### 4. Ball

- **3** — A smooth orange sphere with a black line texture. No spin, or spin
  that is invisible.
- **6** — Correct hue and value, an eight-panel layout, seams that read as
  recessed with some shading, a pebble texture at roughly the right scale, and
  visible backspin. But: the pebble is albedo-only or slightly oversized; the
  sheen is a single circular highlight with no grazing-angle gain; the channels
  are pebbled through; no motion blur; the shadow is constant.
- **9** — Pebble as sub-pixel roughness/normal detail producing a grainy,
  stippled specular band that never aliases and fades correctly with distance.
  Smooth recessed channels of correct width with a bright lip, a dark core and
  AO on the shaded side. Broad tacky sheen with a hard leather rim at the
  silhouette. Correct hue restraint. Warm floor bounce on the underside. Legible
  2.5–4 rev/s backspin, per-object motion blur of ~0.3 diameters at shot speed,
  and a shadow whose penumbra and intensity track height.

### 5. Hoop & net

- **3** — A torus, a transparent quad, and a static cone of thick lines. Nothing
  moves; the ball passes through the net.
- **6** — A shaded rim of correct dimensions, glass with a tint and a specular,
  a net with the right strand count and an hourglass hint, and a simulated net
  that reacts to the ball. But: no rim paint wear, no visible hardware, no glass
  edge refraction, strands too thick, and the net settles too fast or too slow.
  The rim does not flex.
- **9** — Rim with bare-metal wear on the front third, net-burn at the twelve
  attachment points, chipped edges, full visible hardware and breakaway
  assembly. Glass with path-length tint, a real refracting 38 mm edge band, a
  darkened and shifted crowd behind it, and both broad and hard specular
  populations. Net at sub-pixel-safe strand thickness with correct alpha
  coverage, soiled tonal gradient, discernible diamond lattice, hourglass waist
  and gravity sag. On a make: balloon ahead of the ball, hem flip above/near the
  rim plane, 2–3 decaying oscillations over 0.35–0.6 s, asymmetric on off-centre
  entries, zero interpenetration, visible rim flex with a single overshoot.

### 6. Arena & crowd

- **3** — A dark backdrop with a billboard grid of identical figures, or
  nothing at all behind the court.
- **6** — A bowl with a rake, seats, varied crowd silhouettes and some
  desynchronised motion, LED boards that glow, and a courtside layer. But: the
  crowd is one stop too bright, the grid is detectable if you look, the LED
  boards do not light anything, there is no depth haze, and courtside is
  furniture-free.
- **9** — Bowl 2.5–4 stops down with fully legible architecture at that level:
  rake, aisles, vomitories, suite band, railings. Depth haze separating far
  stands from near. Ceiling truss and light banks visible on tilt. Crowd with no
  detectable grid or repetition, overlapping occlusion, eight-plus colour
  clusters, unresolved faces, empty seats, scattered phone-screen points, and
  desynchronised motion. LED ribbons at 190–250 with visible pixel structure,
  content, bloom, and measurable coloured spill onto apron and front rows.
  Individually posed courtside front rows receiving the same rim light as the
  players, plus scorer's table, bench, camera operators, occasional flashes, and
  a hardwood apron that reflects all of it.

### 7. Camera & composition

- **3** — A 75° FOV camera welded to the ball, 6 m up, in a landscape
  composition squeezed into portrait. Action in a horizontal band mid-frame.
- **6** — Correct portrait framing with the hoop and handler in sensible places,
  a reasonable 55–65° FOV, and smoothing on the follow. But: the focal length is
  a little wide, the camera has no lead, DOF is absent or applied uniformly, and
  the composition doesn't use the vertical axis for the shot arc.
- **9** — 38–55° vertical FOV with correct perspective compression (near/far
  sideline ratio ≤ 3.2:1, defender at 80–90% of handler height). Eyeline camera
  height with a 3–12° pitch. Rim and handler at the specified frame fractions
  with HUD safe areas respected. Ball flight is a vertical arc through the tall
  frame. Critically-damped follow with 180–400 ms settle, velocity lead, one
  clean overshoot on hard changes, and sub-0.12° handheld micro-noise. Racked
  focus on the handler with a 120–250 ms lag, 8–22 px CoC on the crowd, gentle
  near-floor blur, clean silhouette edges, and structured bokeh on point
  highlights.

### 8. Post & grade

- **3** — Raw linear output clamped to white. Saturated colours surviving to
  255. Either no bloom or a single huge gaussian haze over the whole frame. No
  vignette, no grain, banded gradients.
- **6** — A filmic tonemapper, threshold bloom, a vignette and grain. But: bloom
  is single-radius and lifts the crowd, the grade is a global saturation/
  contrast push rather than a shadow/highlight split, the vignette is circular
  and too strong for a 19.5:9 frame, and grain is monochrome and scales with
  render resolution.
- **9** — ACES-family curve with a lifted toe, long shoulder and highlight
  desaturation. Multi-mip bloom from countable sources, wide-and-faint, with a
  total mean lift ≤ 4 units and ≤ 3 units on the crowd. Cool shadows at 4–10%
  and warm highlights at 3–8%, measurable but unnameable by eye. Elliptical
  aspect-matched vignette at 10–22% with a wide falloff and slight
  desaturation. Chromatic, luminance-weighted, resolution-independent,
  frame-animated grain at 1.5–4 units. Radial CA under 2 px at the corners,
  excluded from HUD. No sharpen halos, no flares, no wash, no god rays over the
  court.

### 9. Motion & contact

- **3** — Sliding feet, floating shadows, snapping poses, a static net, a rigid
  rim, and a ball that neither spins nor casts a shadow.
- **6** — Planted feet with contact shadows, anticipation on the shot, a
  simulated net, and secondary motion on the jersey. But: some skating on hard
  cuts, contact AO missing under the sole, the rim doesn't flex, the shorts are
  conformal, and the follow-through does not hold.
- **9** — Every planted foot meets its shadow and its reflection with zero gap,
  under 15 mm of drift, coplanar within 2 mm, with 35–55% contact AO rising to
  70–85% at 150 mm. Weight shift with an asymmetric hip line and a compensating
  spine. Wind-up on every impulsive action, a held follow-through with an
  earlier off-hand release, deceleration lean and settle. Secondary motion on
  hem, shorts, hair and head. Hand contacts the ball without gap or
  interpenetration. Rim flexes 2–8 mm and recovers with one overshoot; board
  shakes and transmits to net and padding; the net never intersects the ball.

### 10. Frame cohesion

The tenth category is the one that catches frames that pass every checklist and
still look wrong. It asks a single question: **does this frame read as one
photograph of one room, or as a collection of separately-authored objects
composited together?**

- **3** — Objects clearly come from different worlds. The players are lit
  differently from the floor. The crowd is a backdrop rather than a room the
  court sits inside. Nothing occludes, reflects or shadows anything else.
  Scale relationships are visibly off — the ball is too big, the rim is too
  small, the players are too short for the court.
- **6** — Shared lighting and consistent scale. Everything shadows the floor and
  the floor reflects everything. But there is still a slight "asset library"
  feel: the noise and detail frequencies differ between surfaces (a very
  detailed floor next to a plain jersey, or a highly-detailed ball next to a
  smooth rim), the grade sits on top of the image rather than in it, and one
  element — usually the crowd or the net — clearly belongs to a lower budget
  than the rest.
- **9** — Uniform detail frequency across every surface: no element is
  conspicuously the most or least detailed thing in the frame. Every object
  shadows, occludes, reflects and colour-bleeds onto every other. The grade and
  the grain sit across the whole image as one photographic layer. Scale is
  correct everywhere and cross-checkable (hand spans the ball, ball fits the rim
  with the correct clearance, players fill the correct fraction of the 50 ft
  width). Cover the labels and the frame is simply "a photograph of a basketball
  game," with no object your eye snags on.

---

## Appendix A — Quick-fail checklist

Run this before requesting review. Any **YES** means do not submit.

1. Is the crowd within 2 stops of the court? → YES = fail
2. Does any player cast exactly one shadow? → YES = fail
3. Is there any 200 × 200 px region with a standard deviation under 6? → YES = fail
4. Can you count individual pebbles on the ball at gameplay distance? → YES = fail
5. Is the net a static cone, or drawn thicker than 2 px at gameplay distance? → YES = fail
6. Does a specular highlight change when it crosses a painted line? → YES = fail
7. Is the floor free of scuffs and board-to-board tone variation? → YES = fail
8. Is there a visible gap between any planted sole and its shadow? → YES = fail
9. Is the crowd on a detectable grid? → YES = fail
10. Are saturated colours reaching 255 without desaturating? → YES = fail
11. Is the camera FOV above 65° vertical? → YES = fail
12. Is the rim uniformly orange with no wear? → YES = fail
13. Does the wood surface stop at the boundary line? → YES = fail
14. Does skin lack a warm terminator band? → YES = fail
15. Does the jersey have zero specular response? → YES = fail

## Appendix B — Constants this rubric depends on

All geometry figures here derive from `src/core/Constants.ts`. If a number in
this document disagrees with `Constants.ts`, `Constants.ts` wins and this
document is the bug. Quality-dependent criteria (bloom mips, DOF, motion blur,
crowd count, net iterations, anisotropy) are gated on `src/core/Quality.ts`
tiers and must be evaluated **at the tier the capture was taken at**. State the
tier with every submitted frame; a `low`-tier capture is not exempt from §1,
§2.5, §5.3, §7 or §8.2–8.5, all of which are essentially free.

## Appendix C — Review report format

Every critic report must contain, per category: a score, the single highest-
value fix, and the specific measurement that justified the score. "The lighting
feels flat" is not a review. "Court mean 118, crowd mean 96, ratio 0.3 stops
against a 2.5–4 stop requirement (§1.1) — raise bowl attenuation or drop crowd
ambient" is a review.
