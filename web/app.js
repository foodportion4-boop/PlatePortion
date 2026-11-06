const $ = id => document.getElementById(id);
const form = $('form'), photo = $('photo'), depth = $('depth');
const demo = $('demo');
demo.addEventListener('change',()=>{if(demo.checked){depth.value='';$('depth-name').textContent='None selected';}});
depth.addEventListener('change',()=>{if(depth.files.length)demo.checked=false;});
let speechText = '';
const synth = window.speechSynthesis;
function node(tag, text, cls='') { const e=document.createElement(tag); e.textContent=text; if(cls)e.className=cls; return e; }
function sayStatus(s){$('status').textContent=s;}
function renderMetrics(target, data, suffixes) {
  const grid=node('div','', 'metric-grid');
  for(const [key,label,unit] of suffixes){if(data[key]==null)continue;const box=node('div','', 'metric');box.append(node('strong',`${data[key]}${unit}`),node('span',label));grid.append(box);}
  target.append(grid);
}
function render(data) {
  const p=data.plate, portion=data.portion;
  $('reading').replaceChildren();$('shares').replaceChildren();$('depth-results').replaceChildren();
  if(p){speechText=p.text || ''; $('reading').append(node('p',speechText)); $('reading-indicator').textContent=p.status==='ok'?'PLATE ANALYZED':p.status==='plate_failure'?'RETAKE SUGGESTED':'LOW CONFIDENCE'; $('reading-indicator').classList.toggle('fail',p.status!=='ok');
    const shares=p.portion_shares||[];
    if(shares.length){for(const share of shares){const row=node('div','', 'share-row');const track=node('div','', 'track');const fill=node('div','', 'fill');fill.style.width=`${Math.min(share.plate_area_percent,100)}%`;track.append(fill);row.append(node('strong',share.label),track,node('output',`${share.plate_area_percent}%`));$('shares').append(row);}}
    else $('shares').append(node('p','No reliable food regions to measure.', 'placeholder'));
    if(data.overlay){$('visual').src=data.overlay;$('visual').alt='Plate analysis overlay. The region labels and clock positions are also described in text below.';$('visual').hidden=false;$('image-empty').hidden=true;$('visual-caption').textContent='Image overlay. The text reading and area shares provide the same essential findings.';}
  } else {speechText=''; $('reading').append(node('p','Plate model is not installed yet. See the setup guide for the saved checkpoint.', 'placeholder'));$('reading-indicator').textContent='MODEL NEEDED';$('reading-indicator').classList.add('fail');$('shares').append(node('p','Plate model needed for area shares.', 'placeholder'));}
  if(portion){const illustrative=portion.status==='illustrative';$('depth-results').append(node('p',illustrative?'ILLUSTRATIVE RGB DEMO — assumed heights and densities; no depth was measured.':portion.status==='estimated'?'Depth geometry passed its reliability check.':'Depth geometry was not reliable for this capture.', 'minor'));
    if(illustrative && data.demo_depth_png){const dl=node('a','Download synthetic 16-bit depth PNG · demo only','demo-download');dl.href=data.demo_depth_png;dl.download='synthetic_depth_DEMO_ONLY.png';$('depth-results').append(dl);}
    if(portion.volume_cm3!=null)renderMetrics($('depth-results'),portion,[['volume_cm3',illustrative?'ILLUSTRATIVE VOLUME':'ESTIMATED VOLUME',' cm³']]);
    if(portion.nutrition){$('depth-results').append(node('p',illustrative?'Illustrative mass and energy from assumed density':'Geometric calibration · approximate', 'minor'));renderMetrics($('depth-results'),portion.nutrition,[['mass','Mass',' g'],['calories','Calories',' kcal'],['fat','Fat',' g'],['carb','Carbohydrate',' g'],['protein','Protein',' g']]);}
    if(portion.direct){$('depth-results').append(node('p','Direct RGB-D model · separate prediction', 'minor'));renderMetrics($('depth-results'),portion.direct,[['mass','Mass',' g'],['calories','Calories',' kcal'],['fat','Fat',' g'],['carb','Carbohydrate',' g'],['protein','Protein',' g']]);}
    if(portion.reason)$('depth-results').append(node('p',portion.reason,'minor'));
    if(!portion.nutrition&&!portion.direct)$('depth-results').append(node('p','Calibrated nutrition or direct model checkpoint is not installed.','minor'));
  }else $('depth-results').append(node('p','Add a matching 16-bit depth PNG for numeric estimates.', 'placeholder'));
  const warning=$('warnings');warning.replaceChildren();warning.hidden=!(data.warnings||[]).length;for(const w of data.warnings||[])warning.append(node('p',w));
  $('speak').disabled=!speechText || !synth; $('pause').disabled=$('speak').disabled; $('stop').disabled=$('speak').disabled;
  if(!synth)$('audio-status').textContent='This browser has no speech synthesis. Use a browser with built-in speech, or read the text above with your screen reader.';
}
photo.addEventListener('change',()=>{$('photo-name').textContent=photo.files[0]?.name||'No image selected';if(photo.files[0]){$('visual').src=URL.createObjectURL(photo.files[0]);$('visual').hidden=false;$('image-empty').hidden=true;$('visual').alt='Selected plate photo, before analysis';$('visual-caption').textContent='Selected photo. Choose Analyze plate to see the overlay.';}});
depth.addEventListener('change',()=>{$('depth-name').textContent=depth.files[0]?.name||'None selected';});
form.addEventListener('submit',async e=>{e.preventDefault();if(!photo.files.length){sayStatus('Choose a plate photo first.');photo.focus();return;}if(synth)synth.cancel();$('analyze').disabled=true;sayStatus('Analyzing your plate. This may take several seconds on first run.');try{const res=await fetch('/api/analyze',{method:'POST',body:new FormData(form)});const data=await res.json();if(!res.ok)throw Error(data.error||'Analysis failed');render(data);sayStatus('Analysis finished. Review the reading, portion panels and any warnings.');$('reading').setAttribute('tabindex','-1');$('reading').focus();}catch(err){sayStatus(`Could not analyze: ${err.message}`);}finally{$('analyze').disabled=false;}});
$('speak').addEventListener('click',()=>{if(!synth||!speechText)return;synth.cancel();const utter=new SpeechSynthesisUtterance(speechText);utter.rate=Number($('rate').value);utter.onstart=()=>{$('audio-status').textContent='Reading aloud.';};utter.onend=()=>{$('audio-status').textContent='Readout finished.';};utter.onerror=()=>{$('audio-status').textContent='Speech could not start. Use your screen reader to read the text above.';};synth.speak(utter);});
$('pause').addEventListener('click',()=>{if(!synth)return;if(synth.paused){synth.resume();$('pause').textContent='Pause';}else{synth.pause();$('pause').textContent='Resume';}});
$('stop').addEventListener('click',()=>{if(synth)synth.cancel();$('pause').textContent='Pause';$('audio-status').textContent='Readout stopped.';});
fetch('/api/health').then(r=>r.json()).then(d=>{const s=d.models;sayStatus(s.plate?'Plate model ready. Choose a photo.':'Website ready. Plate checkpoint missing; see the setup guide.');}).catch(()=>sayStatus('Could not reach the local analysis service. Check that the server is running.'));
