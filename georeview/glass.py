"""LiquidGlass overview, isolated from Streamlit's analytical controls."""

COMPONENT = dict(
    name='georeview_liquidglass',
    html='''
<section class="scene" aria-label="Dataset overview">
  <canvas class="backdrop" aria-hidden="true"></canvas>
  <div class="glass">
    <div class="heading"><div><div class="brand">GeoReview India</div>
      <h1>Business review atlas</h1><p class="context"></p></div>
      <span class="status"></span></div>
    <div class="statistics" aria-label="Dataset totals">
      <div><strong class="businesses"></strong><span>Businesses</span></div>
      <div><strong class="reviews"></strong><span>Reviews</span></div>
      <div><strong class="cities"></strong><span>Cities</span></div>
    </div>
  </div>
</section>
''',
    css='''
.scene {position:relative;min-height:270px;padding:24px;box-sizing:border-box;
  font-family:Inter,system-ui,sans-serif;color:#172b3d;isolation:isolate;}
.backdrop {position:absolute;inset:0;width:100%;height:100%;border-radius:28px;z-index:0;}
.glass {position:relative;z-index:1;padding:26px 30px;border-radius:24px;
  background:rgba(255,255,255,.72);border:1px solid rgba(255,255,255,.9);
  box-shadow:0 12px 32px #28446816,inset 0 1px 1px white;}
.heading {display:flex;justify-content:space-between;gap:20px;align-items:flex-start;}
.brand {font-size:14px;font-weight:650;color:#245c72;margin-bottom:10px;}
h1 {font-size:clamp(27px,4vw,42px);font-weight:650;letter-spacing:-1.5px;margin:0 0 9px;line-height:1.12;}
p {font-size:14px;line-height:1.5;max-width:540px;margin:0;color:#39546a;}
.status {font-size:12px;line-height:1.4;padding:7px 12px;border:1px solid #aabccc;
  border-radius:30px;white-space:nowrap;background:#ffffffb8;color:#294458;}
.statistics {display:flex;gap:44px;margin-top:23px;}
.statistics div {display:flex;align-items:baseline;gap:9px;}
.statistics strong {font-size:26px;font-weight:650;letter-spacing:-.8px;}
.statistics span {font-size:12px;color:#39546a;}
@media(max-width:640px) {.scene{padding:13px}.glass{padding:20px}.heading{flex-direction:column;gap:12px}
  .statistics{gap:20px;flex-wrap:wrap}.statistics div{display:block}.statistics span{display:block;margin-top:4px}}
@media(prefers-contrast:more) {.glass{background:white!important;border:2px solid #172b3d}}
''',
    js='''
export default function ({parentElement, data}) {
  const root = parentElement.querySelector('.scene');
  const panel = root.querySelector('.glass');
  for (const key of ['context', 'status', 'businesses', 'reviews', 'cities']) {
    root.querySelector('.' + key).textContent = data[key];
  }
  const canvas = root.querySelector('.backdrop');
  let instance, disposed = false;
  const paint = () => {
    const width = Math.max(root.clientWidth, 1), height = root.clientHeight;
    canvas.width = width; canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#dbeaf4'; ctx.fillRect(0, 0, width, height);
    for (const [x,y,r,color] of [[.15,.1,.6,'#8fc8c6'],[.9,.7,.65,'#b4b9ed'],[.65,0,.4,'#f3dbcb']]) {
      const gradient = ctx.createRadialGradient(x*width,y*height,0,x*width,y*height,r*width);
      gradient.addColorStop(0,color); gradient.addColorStop(1,color+'00');
      ctx.fillStyle=gradient; ctx.fillRect(0,0,width,height);
    }
    instance?.markChanged(canvas);
  };
  paint();
  const observer = new ResizeObserver(paint);
  observer.observe(root);
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const contrast = matchMedia('(prefers-contrast: more)');
  const stop = () => { instance?.destroy(); instance = null; panel.style.background = ''; };
  const preferenceChanged = () => { if (reduced.matches || contrast.matches) stop(); };
  reduced.addEventListener('change', preferenceChanged);
  contrast.addEventListener('change', preferenceChanged);
  if (!reduced.matches && !contrast.matches) {
    import('https://cdn.jsdelivr.net/npm/@ybouane/liquidglass@1.0.3/dist/index.js')
      .then(async ({LiquidGlass}) => {
        if (disposed || reduced.matches || contrast.matches) return;
        panel.dataset.config = JSON.stringify({cornerRadius:24,blurAmount:.18,refraction:.35,
          chromAberration:.015,edgeHighlight:.08,shadowOpacity:.12,floating:false});
        const created = await LiquidGlass.init({root,glassElements:[panel]});
        if (disposed || reduced.matches || contrast.matches) { created.destroy(); return; }
        instance = created;
        root.dataset.renderer = 'webgl';
      })
      .catch(error => { root.dataset.renderer = 'fallback'; console.warn('LiquidGlass fallback', error); });
  }
  return () => { disposed=true;observer.disconnect();stop();
    reduced.removeEventListener('change',preferenceChanged);
    contrast.removeEventListener('change',preferenceChanged); };
}
''',
)


def overview(data, synthetic, renderer):
    renderer(data={
        'context': ('Explore fictional businesses and simulated reviews across Indian cities.' if synthetic
                    else 'Explore historical review patterns among geographically validated Indian businesses.'),
        'status': 'Synthetic demonstration' if synthetic else 'Historical research data',
        'businesses': f'{len(data):,}', 'reviews': f'{int(data.review_count.sum()):,}',
        'cities': f'{data.loc[data.city.ne("Unknown city"), "city"].nunique():,}',
    }, key='glass_overview')
