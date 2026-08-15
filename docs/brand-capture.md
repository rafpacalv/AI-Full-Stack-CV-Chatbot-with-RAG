# leadtech.com — brand capture and computed-style audit

Captured 2026-08-15 with Playwright/Chromium (headless, 1600x1000, `networkidle` + 3s settle,
full-page scroll before capture). Pages audited: `/`, `/about-us`, `/work-with-us`, `/contact`.
Every figure below comes from `getComputedStyle` over the live DOM, from a pixel histogram of the
full-page PNGs, or from the site's own stylesheet
(`https://leadtech.com/assets/c819ff5b4e3014b573fdb3b11672ee57.css`, 425 KB).

**No blocks encountered.** HTTP 200, no Cloudflare challenge, no bot wall. A cookie banner
(`.m-cookies-policy`) appears bottom-of-page; it was dismissed via its `Accept` link before every
screenshot, so nothing is obscured.

---

## 0. Headline corrections

Three assumptions worth fixing before anything else:

1. **The site is not dark. It is overwhelmingly light.** `body { background-color: #FFFFFF }` on
   all four pages. White alone is 51–61 % of every full-page screenshot; pixels brighter than
   luminance 180 are 66–80 % of the page. Dark pixels (<70 luminance) are 10–17 %.
2. **`#140C29` is not a Leadtech brand colour.** It exists in the CSS (16 occurrences) but
   17 of the 26 rules using it or its neighbours (`#140A28`, `#130A29`) are scoped to
   `.adventure-2022` — a one-off gamified campaign microsite. It appears **zero times** in the
   computed styles of the home, about-us, work-with-us or contact pages. The site's actual dark
   tone is pure black `#000000` plus a near-black ink `#262627`.
3. **Mint `#00FFC6` is real and is used as a *fill*, not a hairline.** But it is one member of a
   rotating accent set, not a permanent site-wide accent — see §3.

---

## 1. Palette table

Dominance = share of pixels in the 1600 px-wide full-page screenshot of that page.

| Hex | Name | Where it is used | Dominance |
|---|---|---|---|
| `#FFFFFF` | Page white | `body` background on every page; the default canvas; text on all black/dark panels; the "bright" button variant `.m-buttons-primary--bright` | **51 % (home) / 61 % (about) / 58 % (work) / 61 % (contact)** — the dominant colour by a wide margin |
| `#000000` | Pure black | Big content cards (`.m-hightligth-darkbox`, `.m-panel-slideText-buttonbox`), footer "Stay tuned" block (`.m-footer-top-left`), the fixed 65×65 hamburger tile, the full-screen nav overlay | 3.7–6.1 % |
| `#262627` | Ink / near-black | `body` text colour, `.m-title-regular` headings, inner-page jumbotron strip (`.m-jumbotron-single-bg`), footer legal bar (`.m-footer-bottom`), `.m-hightligth-descriptiondark` card, the "bcn 30ºc" hero chip | 1.7–7.2 % |
| `#00FFC6` | **Mint** (brand accent) | Default `.m-buttons-primary` fill; entrepreneur card `.m-hightligth-ligthbox`; discipline-page intro panel `.m-panel-slideText-slide-title-txt`; `.m-panel-box`; "Barcelona HQ" chip; all four footer social SVG fills; the `.m-title-regular-icon-tech:before` glyph | 0.3 % (contact) → 1.2 % (home) → **5.5 % (work-with-us)** |
| `#D1F2FF` | Pale sky | Footer link panel `.m-footer-top-right`; wide quiet section backdrops (`.m-panel-bg-light:before`, `.m-panel-bg-right-column`, `.m-jumbotron-hero-bg`) | 3.3–12.8 % — **the second-largest colour field on the site, larger than mint everywhere** |
| `#F7B53E` | Mustard | Social-responsibility band `.l-text-background-image-wrapper`; that band's button (radius forced to 0); `.m-panel-section-header-mustard` heading + 3 px underline; `.asideNav__item:nth-child(3)` | ~1.1 % on every page (it is in the shared CSR section) |
| `#F54F81` | Hot pink | About-us "what we offer" column block and its button override; `.m-panel-columns-background...what-we-offer` | 4.2 % on about-us, 0 elsewhere |
| `#2CDE97` | Emerald | One of five randomised hero backgrounds (`.m-animations-bg-leadtheball`); also the "Backend" discipline colour | **14.9 % of the home page — but only because the hero happened to draw this theme.** Not stable. |
| `#FFB459` | Amber (hero) | `.m-animations-bg-leadthesun` hero variant | 0 or ~15 % depending on the draw |
| `#3369E7` | Royal blue | `li#idBall` hero prop; "Customer service" discipline colour | <0.3 % |
| `#50E3C2` | Soft mint | **Hover/link colour only** — `.m-header-nav .m-nav-links a:hover`, and the footer `Privacy Policy` link | trace |
| `#15856C` | Deep teal | Footer social icon **hover** fill only | trace |
| `#44DE97` | Green (secondary) | `.m-panel-section-header-green` heading + underline, `.m-panel-item-icon-green`, `.stakeholders__card--secondary`, `.asideNav__item:nth-child(4)` | small, CSR/about pages |

### The 14-colour discipline palette

A separate closed set, used **only** to colour-code the 14 job disciplines. Each appears as a
49×49 icon tile background (`.c-*`) and as a 19 px bold label colour (`.c-font-*`), and as the
whole-page theme on that discipline's `/work-with-us/<slug>` page via `body.<slug>`:

`#FF4F81` UX · `#8E44E7` Frontend · `#E74444` SEM · `#2CDE97` Backend · `#FFC168` Project mgmt ·
`#B84692` Finance · `#1BC7D0` BI · `#FF6C5E` SEO · `#3369E7` Customer service · `#04AEFF` Social
media · `#0060B4` Content · `#8687E9` QA · `#F99F61` Sysadmin · `#85D0E9` HR

Text on these tiles is `#262627` except Finance and BI, which use `#fff`.

---

## 2. Dominant background, verified

| Page | body background | white | light (>180 lum) | dark (<70 lum) |
|---|---|---|---|---|
| `/` | `#FFFFFF` | 51.0 % | 66.3 % | 10.1 % |
| `/about-us` | `#FFFFFF` | 61.5 % | 73.1 % | 13.2 % |
| `/work-with-us` | `#FFFFFF` | 58.3 % | 77.8 % | 11.9 % |
| `/contact` | `#FFFFFF` | 61.2 % | 80.1 % | 17.0 % |

The dark passages are **hard-edged black rectangles sitting on white**, not a dark page with light
cards. Contrast is maximal and binary: `#000` / `#262627` blocks against `#FFF`, no mid-greys, no
elevation ramp.

The hero is the one exception — a full-viewport 1600×1000 saturated colour field. It draws at
random from five themes defined in CSS:

| class | background | prop colour |
|---|---|---|
| `.m-animations-bg-leadthesun` | `#FFB459` | `#2DDE98` |
| `.m-animations-bg-leadtheball` | `#2CDE97` | `#3369E7` |
| `.m-animations-bg-leadcreativity` | `#04AEFF` | `#FF6B5D` |
| `.m-animations-bg-leadbarcelona` | `#FF4F81` | `#04AEFF` |
| `.m-animations-bg-chuckleadyou` | `#FFF` | `#F3AB27` |

So **the single biggest colour block on the home page is deliberately non-deterministic**. Any
"the hero is green" reading of leadtech.com is an artefact of one page load.

---

## 3. Accent usage — what mint is actually for

This is the part worth reading carefully, because the answer is "both, in a specific way".

I scanned every visible element on all four pages for any colour in hue 120–200° with saturation
≥ 40 %, across `backgroundColor`, `color`, `borderColor`, `fill` and `stroke`. Mint `#00FFC6` was
the only such colour that recurred site-wide. Its roles, by pixel area:

**Mint is used as a solid fill on medium-sized blocks — never as a hairline, underline, or border.**

Measured mint elements (home / work-with-us):

| Element | Size | Role |
|---|---|---|
| `.m-hightligth-ligthbox` — "for Entrepreneurs" card | 527 × 276 px (145 k px²) | **full card fill**, body text `#262627` on top |
| `.m-panel-slideText-slide-title-txt` — intro panel, work-with-us | 583 × 213 px (124 k px²) | **full panel fill**, black text |
| `a.m-buttons-icon-linkdin` — "SEE OPEN POSITIONS ON LINKEDIN" | 282 × 51 px | **button fill**, black text |
| `a.m-footer-linkgreen` — "LET'S SEE YOUR STRATEGY" | 256 × 51 px | **button fill** on a black card |
| `.m-hightligth-names-bcn` — "Barcelona HQ" chip | 200 × 57 px | **label chip fill** |
| `button#button-contact-submit` — "SEND" | 119 × 39 px | **button fill** |
| footer social SVGs (fb/tw/li/ig) | 4 × ~32 × 32 px | **icon fill** on a black block |
| `.m-title-regular-icon-tech:before` | 2em glyph | **icon glyph colour** |

Notice what is *absent*: mint is **never** a border colour (the computed-style pass found
**zero** non-transparent borders anywhere on any of the four pages), never an underline, never a
link colour in running text, never a heading colour, never small text. There is no
`border: 1px solid #00FFC6` anywhere in the brand stylesheet.

Three further characteristics:

- **Mint always carries black text, never white.** `.m-buttons-primary { background:#00FFC6;
  color:#000 }`. Mint is treated as a *light* surface — it is L≈50 %, luminance ~205 — and it
  pairs with `#000`. Putting white text on mint would break the pattern completely.
- **Mint reads brightest against black, not against white.** The two places it is most striking
  are the mint button inside the black "for Investors" card and the mint social icons inside the
  black "Stay tuned" block. Against the white body it is a soft pastel; against black it snaps.
- **Mint is not exclusive.** It is the *default* value of `.m-buttons-primary`, but sections
  override it wholesale: the social-responsibility band swaps every button to `#F7B53E` **and
  sets `border-radius: 0`**; the about-us "what we offer" band swaps them to `#F54F81`, also
  square. On `/about-us` mint has essentially zero fill presence — pink `#F54F81` (4.2 %) takes
  the accent role instead. Mint's share ranges 0.3 % → 5.5 % across pages.

Softer mints do exist and are **strictly interactive-state colours**: `#50E3C2` is the nav-link
hover colour and the `Privacy Policy` link colour, `#15856C` is the social-icon hover fill. Mint
proper (`#00FFC6`) is never a hover colour, and hover never changes it.

**Summary in one line:** mint is a *surface* colour — it fills buttons, chips, and one card or
panel per screen — and is deliberately used at roughly 1–5 % of page area, always with black
text, never as a stroke, outline, or text colour.

---

## 4. Button specs

Base rule, verbatim from the stylesheet:

```css
.m-buttons-primary, .m-buttons-icon-linkdin, .m-footer-linkbright, .m-footer-linkgreen {
  font-size: 1em;            /* = 13px, html is 10px, body 13px */
  background: #00FFC6;
  color: #000;
  border-radius: 3px;
  font-weight: 500;
  margin-top: .5em;
  display: inline-block;
  text-transform: uppercase;
  letter-spacing: .6px;
  padding: 16px 40px;
  transition: all 0.2s;
}
.m-buttons-primary--bright, .m-footer-linkbright { background: #fff; }
.m-buttons-primary--green,  .m-footer-linkgreen  { background: #00FFC6; }
.m-buttons-primary--contact { padding: 10px 42px; border: none; }
.m-buttons-icon-linkdin     { background: #00FFC6; color: #000; padding: 16px; }
```

Computed, as rendered:

| Button | bg | text | border | radius | padding | font | size |
|---|---|---|---|---|---|---|---|
| "SEE OPEN POSITIONS ON LINKEDIN" | `#00FFC6` | `#000000` | none | 3 px | 16 px | Roboto 500 / 13 px, uppercase, ls .6 px | 282 × 51 |
| "LET'S SEE YOUR STRATEGY" (on black card) | `#00FFC6` | `#000000` | none | 3 px | 16 / 40 px | same | 256 × 51 |
| "SHOW US YOUR IDEAS" (on mint card) | `#FFFFFF` | `#000000` | none | 3 px | 16 / 40 px | same | 224 × 51 |
| "SEE JOBS" (CSR band) | `#F7B53E` | `#000000` | none | **0 px** | 16 / 40 px | same | 143 × 51 |
| "SEE OPEN POSITIONS IN LINKEDIN" (about-us) | `#F54F81` | `#000000` | none | **0 px** | 16 / 20 px | same | 263 × 51 |
| "SEND" (contact form) | `#00FFC6` | `#000000` | none | 3 px | 10 / 42 px | same | 119 × 39 |

Notes:

- **Every button on the site has black text.** No white-on-colour buttons exist.
- **No borders, no box-shadow, no gradient** on any button.
- **Hover does nothing.** I hovered all four home-page CTAs and diffed the computed styles: zero
  change in background, colour, border, radius, shadow or background-image. The only interaction
  feedback is `:focus`/`:active` → `transform: scale(.95)` — a 5 % press-in. The `transition: all
  0.2s` is there purely for that scale.
- **The mint↔white swap is contextual, not hierarchical.** On a mint card the CTA is white; on a
  black card the CTA is mint. Both are "primary". The rule is *maximum separation from the card
  behind it*, not primary-vs-secondary.
- Button height is a consistent 51 px (39 px for the compact form variant).

---

## 5. Card / panel specs

| Component | Background | Text | Padding | Radius | Border | Shadow |
|---|---|---|---|---|---|---|
| `.m-hightligth-ligthbox` (mint card) | `#00FFC6` | `#262627` | `30px 40px 40px` | **0** | none | none |
| `.m-hightligth-darkbox` (black card) | `#000000` | `#FFFFFF` | `30px 40px 30px 140px`, ls .3px | **0** | none | none |
| `.m-hightligth-descriptiondark` | `#262627` | `#FFFFFF` | `0 25px`, max-w 350, h 200 | **0** | none | none |
| `.m-panel-slideText-buttonbox` | `#000000` | `#FFFFFF` | `20px 40px` | **0** | none | offset black `:before` block at `left:66px` |
| `.m-panel-box` | `#00FFC6` | inherit | `20px` | **0** | none | mint `:before` bleed block |
| `.m-panel-slideText-slide-title-txt` | `#00FFC6` | `#000` | `30px 40px 40px 40px` | **0** | none | none |
| `.m-footer-top-left` (Stay tuned) | `#000000` | `#FFFFFF` | `padding-left:30px`, h 260 | **0** | none | none |
| `.m-footer-top-right` (link panel) | `#D1F2FF` | `#000000` | `20px 0`, pulled `margin-top:-100px` | **0** | none | none |
| `.m-footer-bottom` (legal bar) | `#262627` | `#FFFFFF` | h 80 | **0** | none | none |
| `.l-column-background-item-icon` | `#FFFFFF` | — | 49 × 49 | **0** | none | `-10px 10px 0 0 black` |
| `.m-hightligth-names-bcn` (chip) | `#00FFC6` | `#000` | `17px 40px`, w 200 | **0** | none | none |

Structural rules that follow from this:

- **`border-radius: 3px` exists on exactly one thing: buttons.** Every card, panel, chip, tile
  and section is a hard 0 px rectangle. Radius counts across the whole home page: `{3px: 5}`.
- **Zero borders.** The computed-style sweep returned an empty border-colour map on all four
  pages. Separation is done with adjacent solid colour blocks, never with strokes.
- **Effectively zero soft shadows.** The only shadows on brand surfaces are
  `-10px 10px 0 0 black` (hard offset, no blur, on the 49×49 white icon squares) and a
  `0 2px 4px rgba(0,0,0,.16)` on the cookie-accept chrome. No elevation system, no blur.
- **Overlap is the layout device.** Cards are pulled into each other with negative margins
  (`margin-left:-486px`, `margin-right:-590px`, `margin-top:-100px`) so blocks bleed off-grid and
  overlap. The mint card and black card in "down to business" deliberately interlock.
- **No gradients anywhere on brand pages.** `backgroundImage` was `none` on every element in all
  four audits. Gradients exist only inside `.adventure-2022`
  (`linear-gradient(to right,#00cedd,#84ecff)` on campaign-game buttons).

---

## 6. Typography actually observed

Root `html { font-size: 10px }`; `body { font-family: "Roboto", sans-serif; font-size: 13px;
color: #262627 }` — so `1em` on a component = 13 px.

Three families, with clear division of labour:

| Family | Role |
|---|---|
| **leadtech-Bold** (custom, self-hosted) | Display only — hero titles, section titles, nav links. Always the heaviest, always tightly tracked. |
| **Merriweather** (serif, Google) | Sub-headings, pull quotes, card headings, editorial emphasis. Frequently *italic*. This is what gives the site its editorial feel. |
| **Roboto** (Google, 400/500/900) | Everything functional — body copy, buttons, footer, labels. |

Also present: `leadtech-font` (an icon font, `content:"\e901"` etc.) and `leadtoji`.

Observed scale, by measured computed value:

| Role | Family | Weight | Size | Line-height | Letter-spacing | Colour |
|---|---|---|---|---|---|---|
| Hero title `.m-title-hero` | leadtech-Bold | 900 | **80 px** | 65 px | **−1.68 px** | `#FFF` on colour, `#000` on white |
| Home hero greeting `h2#hi` | leadtech-Bold | 900 | 65 px | 65 px | −1.18 px | `#262627` |
| Section title `.m-title-regular` | leadtech-Bold | 900 | **65 px** | 1em | **−1.18 px** | `#262627` |
| Nav overlay links | leadtech-Bold | 500 | 64 px (`calc(4*(1vw+1vh-1vmin))`) | 83.2 px | normal | `#FFFFFF` |
| Hero wordmark ("leadtheball") | leadtech-Bold | 400 | 90 px | 13 px | normal | `#FFFFFF` |
| Footer heading "Stay tuned" | Merriweather | 800 | 40 px | 50 px | normal | `#FFFFFF` |
| Sub-heading `.m-hightligth-subtitle` | Merriweather *italic* | 400 (b → 900) | 30 px | 42.9 px | normal | `#262627` / `#FFF` |
| Card heading ("for Entrepreneurs") | Merriweather *italic* | 400 | 30 px | 42.9 px | normal | `#262627` |
| `.m-subtitle-small` | Merriweather | 400 (b → 900 italic) | 26 px | 1.33em | normal | `#262627` |
| Pull quote `.mention` | Merriweather | 400 | 18.2 px | — | normal | `#262627` |
| Body copy | Roboto | 400 | 16.3 px | — | normal | `#262627` |
| Small body / lists | Roboto | 400 | **14 px** (most frequent size on the page, 24 occurrences) | — | normal | `#262627` |
| Footer nav links | Roboto | 700 | 18 px | — | normal | `#000000` |
| Discipline labels | Roboto | **900** | 19 px | — | normal | one of the 14 codes |
| Buttons | Roboto | 500 | **13 px** | — | **.6 px**, `uppercase` | `#000000` |
| Footer column heads | Roboto | 600 | 14 px | — | normal | `#262627` |

The pattern: **display type is huge (65–90 px) and negatively tracked; UI type is tiny (13–14 px)**.
There is almost nothing in the 20–25 px middle. That gap is the single most characteristic thing
about the typography.

---

## 7. Screenshots

All under `docs/img/brand/`.

**Home**
- [`home-hero.png`](img/brand/home-hero.png) — 1600×1000 above the fold. Full-bleed `#2CDE97`
  hero, white 90 px wordmark, black 65×65 hamburger tile, black "bcn 30ºc" chip, `#3369E7` prop.
- [`home-full.png`](img/brand/home-full.png) — full page, 1600×5344.
- [`home-nav.png`](img/brand/home-nav.png) — top 200 px, nav bar crop.
- [`home-nav-menu.png`](img/brand/home-nav-menu.png) — the nav overlay **open**: pure `#000`
  full-screen, white 64 px leadtech-Bold links, no mint at all.
- [`home-whatwedo.png`](img/brand/home-whatwedo.png) — "what we do" section.
- [`home-discipline-colours.png`](img/brand/home-discipline-colours.png) — the 14-colour
  discipline tile grid, the site's most colour-dense area.
- [`home-cards.png`](img/brand/home-cards.png) — "who are we?" cards/tiles.
- [`home-highlight-boxes.png`](img/brand/home-highlight-boxes.png) — **the key reference shot**:
  mint card + black card overlapping, white CTA on mint, mint CTA on black, mint social icons on
  black, pale-sky footer beginning.
- [`home-cta.png`](img/brand/home-cta.png) — CTA band.
- [`home-cta-button.png`](img/brand/home-cta-button.png) — close crop of a single CTA.
- [`home-footer.png`](img/brand/home-footer.png) — the real footer: black block, `#D1F2FF` link
  panel, white address strip, `#262627` legal bar.

**Inner pages**
- [`about-us-hero.png`](img/brand/about-us-hero.png) / [`about-us-full.png`](img/brand/about-us-full.png)
  — accent role taken by `#F54F81` (4.2 %); mint absent as a fill.
- [`work-with-us-hero.png`](img/brand/work-with-us-hero.png) / [`work-with-us-full.png`](img/brand/work-with-us-full.png)
  — the mint-heaviest page (5.5 %), big `#00FFC6` intro panel with black text.
- [`contact-hero.png`](img/brand/contact-hero.png) / [`contact-full.png`](img/brand/contact-full.png)
  — dark `#262627` form panel with white inputs, mint SEND button, mint "Barcelona HQ" chip.

---

## 8. What a dark app UI should copy from this

leadtech.com is a light site, so a dark app cannot copy it literally. What transfers is the
*discipline*, and these are the concrete rules to lift:

1. **Do not use mint as a large fill in a dark UI.** On leadtech.com mint fills only 0.3–5.5 % of
   the page, and it does so *against white* where it reads as a pastel. Dropped onto a dark
   background at the same coverage it will be several times louder, because mint against `#000`
   is the site's own highest-contrast pairing (it is exactly what they reserve for the one CTA
   inside the black investor card). Budget mint at **1–3 % of the dark viewport, maximum**, and
   keep it to the same element classes they use: primary button fill, one status chip, icons.

2. **Mint always takes black text, never white.** `#00FFC6` with `#000` at Roboto 500 / 13 px /
   uppercase / .6 px tracking is the exact button. White text on mint appears nowhere on the site
   and would fail contrast anyway (mint luminance ≈ 205). If your UI currently has white-on-mint,
   that is off-brand *and* less legible.

3. **Mint is a surface, not a stroke.** There is not one mint border, underline, divider or
   link colour on the entire site. If your UI is using mint for 1 px outlines, focus rings, or
   sparkline strokes, that is the wrong role — move it onto the button and chip fills instead.
   For interactive *states* they step **away** from mint, not into it: `#50E3C2` for link hover,
   `#15856C` for icon hover. Both are duller than the base mint.

4. **Ship square corners.** `border-radius: 3px` on buttons and `0` on absolutely everything else.
   No rounded cards, no pill badges, no 8/12/16 px radius scale. A dark UI copying this should set
   card radius to 0 and button radius to 3 px and stop there.

5. **No shadows, no gradients, no borders.** Zero non-transparent borders on four pages; zero
   gradients; the only shadow idiom is a hard `-10px 10px 0 0 black` offset with no blur. In a
   dark UI, separate panels by **stepping the flat background colour** — `#000` next to `#262627`
   next to a colour block — never with a 1 px grey line or a soft elevation shadow. Their whole
   layout language is adjacent and overlapping solid rectangles.

6. **Use black and near-black as two distinct tokens.** `#000000` is for hero content blocks and
   the nav overlay; `#262627` is for body text, the legal bar and secondary panels. That two-step
   near-black is the closest thing the brand has to a dark surface scale — use `#000` as the app
   canvas and `#262627` as the raised panel, not an indigo.

7. **Steal `#D1F2FF` as the quiet secondary.** It is larger on the page than mint everywhere
   (3.3–12.8 %) and does the work of "this region is a calm supporting block". In a dark UI it is
   too bright for a surface, but it is the right colour for **secondary/informational text and
   icons on black** — it is already a brand colour paired with dark, and it gives you somewhere to
   go that is not mint.

8. **Only one accent per screen, and rotate it by section.** The site never shows mint, mustard
   and pink competing in the same viewport. `/about-us` is pink, the CSR band is mustard,
   `/work-with-us` is mint, and each discipline page recolours its entire theme via `body.<slug>`.
   Give each app view one accent, and if you want variety, vary it per view — do not mix.

9. **Copy the type contrast, not the type size.** Display at 65–80 px weight 900 with −1.2 to
   −1.7 px tracking; UI at 13–14 px. Scaled into an app that becomes roughly a 32–40 px page title
   at weight 800–900 with negative tracking against 13–14 px body — with essentially nothing in
   between. And the serif matters: Merriweather italic at 26–30 px is what makes their cards feel
   editorial rather than SaaS. A dark app that uses one sans at 16/20/24 px will not read as
   Leadtech no matter how correct the mint is.

10. **Buttons do not light up on hover.** No colour, background or shadow change — only
    `transform: scale(.95)` on press. If you want to feel like their site, resist the hover-glow.

---

## Appendix — capture caveats

- The hero background is randomised per page load across five themes (§2), so `home-hero.png` and
  the 14.9 % `#2CDE97` figure in the home-page histogram represent one draw, not a fixed value.
  Six consecutive reloads during the audit returned `leadthesun` (`#FFB459`) each time, while the
  screenshot session drew `leadtheball` (`#2CDE97`) — treat hero colour as variable.
- The home page hero greeting is time-of-day dependent ("hi, good afternoon from leadtech") and
  shows a live Barcelona temperature chip.
- `home-cards.png` and `home-cta.png` were captured at scroll offsets derived from section
  headings; the sections partially overlap in framing.
- Mint-adjacent colours were detected by an HSL filter (hue 120–200°, sat ≥ 40 %, lightness
  20–85 %) across `backgroundColor`, `color`, `borderColor`, `fill` and `stroke`. Mint SVG paths
  are counted once per nested `<g>`/`<path>`, which is why raw counts (19 `fill` hits) exceed the
  four visible social icons — the area figures are the meaningful numbers.
- Pixel-dominance percentages come from a 4× downsample of each full-page PNG, so sub-0.1 %
  colours may be under-reported.
