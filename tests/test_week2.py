import copy
import numpy as np
import pytest
from plateclock.pipeline import load_config, run, placement
from plateclock.language.style_guide import load_style
from plateclock.language.plate_nlg import describe
from plateclock.language.style_lint import lint, require_valid
from plateclock.plate.confidence import decide
from plateclock.plate.group_regions import group_regions, Region, class_groups
from plateclock.plate.geometry import ItemPlacement
from plateclock.plate.evaluation import match_regions, threshold_table
from plateclock.plate.fit_quality import fully_visible, quality
from plateclock.plate.detect_plate import PlateFit
from plateclock.io.class_names import CLASS_NAMES

@pytest.fixture
def cfg(): return load_config()

def test_complete_class_mapping(cfg):
    mapping = class_groups(CLASS_NAMES, cfg)
    assert set(mapping) == set(range(1,104))
    assert mapping[66] == 'staples' and mapping[48] == 'protein'

def test_grouping_and_denominators(cfg):
    labels = np.zeros((20,20), np.uint8)
    labels[0:5,0:5] = 66; labels[0:5,5:10] = 65
    labels[15:20,15:20] = 66; labels[9,9] = 48
    labels[10,10] = 255
    result = group_regions(labels, np.ones_like(labels, dtype=float), np.ones_like(labels, dtype=bool), CLASS_NAMES, cfg)
    assert len(result) == 2  # touching rice/noodles merge; remote rice stays separate
    assert result[0].area_fraction == 50/400
    assert result[0].share == 50/76  # dropped food remains in denominator, ignore 255 does not

@pytest.mark.parametrize('prob,action',[(.249,'omit'),(.25,'hedge'),(.599,'hedge'),(.6,'keep')])
def test_confidence_boundaries(cfg, prob, action):
    assert decide(prob,.1,1,cfg['confidence']).action == action

def test_invalid_probabilities(cfg):
    with pytest.raises(ValueError): decide(float('nan'),.1,1,cfg['confidence'])
    with pytest.raises(ValueError): group_regions(np.ones((2,2)),np.full((2,2),1.1),np.ones((2,2)),CLASS_NAMES,cfg)

def item(hour=12, label='staples', share=.6):
    return ItemPlacement(label,hour,'middle',share,(hour,hour),(hour%12)*30)

def test_nlg_order_hedge_and_summary():
    items = [item(3,'protein',.2),item()]
    text = describe(items,['hedge','keep'])
    assert text == "2 items identified. staples at 12 o'clock, middle. Looks like protein at 3 o'clock, middle."
    assert not lint(text)
    assert not lint(describe(items,['hedge','keep'],'summary'), 'summary')
    assert not lint(describe([item(12,'sauce or soup',.8)],['hedge'],'summary'), 'summary')

@pytest.mark.parametrize('text',[
    "2 items identified. staples at 12 o'clock, middle.",
    "1 item identified. rice at 12 o'clock, middle.",
    "1 item identified. protein at 13 o'clock, middle.",
    "1 item identified. protein at 3 o'clock, outside.",
    "1 item identified. Maybe protein at 3 o'clock, middle.",
    "2 items identified. protein at 3 o'clock, middle. staples at 12 o'clock, middle.",
    "1 item identified. protein at three o'clock, middle.",
    "1 item identified. protein at 3 o'clock, middle, 50 grams.",
    "1 item identified. protein at 3 o'clock, middle!",
    "1 item identified. protein at 3 o'clock, middle..",
])
def test_bad_language(text):
    assert lint(text)

def test_exact_failures():
    s=load_style()
    assert not lint(s['failure']) and not lint(s['empty'])

def test_single_pixel_arc_and_wrap(cfg):
    mask=np.zeros((101,101),bool);mask[10,50]=True
    p=placement(Region('protein',mask,.1,1,.9),(50,50),50,cfg['geometry'])
    assert p.hour == 12 and p.span == (12,12)
    mask[10,49]=True; mask[10,51]=True
    p=placement(Region('protein',mask,.1,1,.9),(50,50),50,cfg['geometry'])
    assert p.span == (12,12)

def test_one_to_one_and_zero_retention():
    m=np.ones((5,5),bool);r=Region('protein',m,1,1,.9)
    assert sum(match_regions([r,r],[r],m)) == 1
    table=threshold_table([.2,.8],[0,1])
    assert table[-1]['precision'] is None
    assert table[10]['precision'] == 1 and table[10]['coverage'] == .5

def test_cropped_ellipse():
    assert not fully_visible(PlateFit((10,10),(100,100),0,1),(100,100))

def test_pipeline_and_no_food(cfg):
    import cv2
    image=np.full((321,321,3),25,np.uint8)
    cv2.circle(image,(160,160),125,(235,235,235),-1)
    class Model:
        def predict(self,image):
            labels=np.zeros(image.shape[:2],np.uint8)
            cv2.circle(labels,(160,100),22,66,-1)
            return labels,np.ones(labels.shape,np.float32)
    result=run(image,Model(),cfg)
    assert result['status']=='ok'
    assert result['items'][0].hour==12
    assert not lint(result['text'])
    cfg=copy.deepcopy(cfg);cfg['confidence']['omit_below']=1;cfg['confidence']['keep_at']=1
    assert run(image,Model(),cfg)['status']=='no_confident_food'

def test_blank_plate_failure(cfg):
    class Model:
        def predict(self,image):return np.zeros(image.shape[:2],np.uint8),np.zeros(image.shape[:2],np.float32)
    result=run(np.zeros((100,100,3),np.uint8),Model(),cfg)
    assert result['status']=='plate_failure' and not lint(result['text'])

def test_lint_blocks_output(monkeypatch, cfg):
    import plateclock.pipeline as pipe
    import cv2
    image=np.full((321,321,3),25,np.uint8);cv2.circle(image,(160,160),125,(235,235,235),-1)
    class Model:
        def predict(self,image):
            labels=np.zeros(image.shape[:2],np.uint8);labels[70:120,140:180]=66
            return labels,np.ones(labels.shape,np.float32)
    monkeypatch.setattr(pipe,'describe',lambda *args:'Uncontrolled output.')
    with pytest.raises(ValueError,match='Style lint failed'):pipe.run(image,Model(),cfg)

def test_tts_contract_with_fake_driver(monkeypatch, tmp_path):
    import sys, types, wave
    from plateclock.audio.tts import save
    settings={}
    class Engine:
        def getProperty(self,key):return [types.SimpleNamespace(id='test-voice')]
        def setProperty(self,key,value):settings[key]=value
        def save_to_file(self,text,path):self.path=path
        def runAndWait(self):
            with wave.open(self.path,'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(22050);f.writeframes(b'\x01\x00'*100)
        def stop(self):pass
    monkeypatch.setitem(sys.modules,'pyttsx3',types.SimpleNamespace(init=Engine))
    monkeypatch.setattr('plateclock.audio.tts.platform.system',lambda:'Linux')
    target=save('Test.',tmp_path/'audio.wav','test-voice',165)
    assert target.stat().st_size>44 and settings=={'voice':'test-voice','rate':165}
    with pytest.raises(ValueError,match='not installed'):
        save('Test.',target,'missing-voice')

def test_cli_synthetic_artifacts(monkeypatch,tmp_path):
    import sys, types, cv2, json
    from PIL import Image
    from plateclock.app.cli import main
    class Model:
        def __init__(self,*args):pass
        def predict(self,image):
            labels=np.zeros(image.shape[:2],np.uint8);cv2.circle(labels,(160,100),22,66,-1)
            return labels,np.ones(labels.shape,np.float32)
    image=np.full((321,321,3),25,np.uint8);cv2.circle(image,(160,160),125,(235,235,235),-1)
    source=tmp_path/'input.png';Image.fromarray(image).save(source)
    monkeypatch.setitem(sys.modules,'plateclock.plate.segment_food',types.SimpleNamespace(FoodSegmenter=Model))
    monkeypatch.setattr(sys,'argv',['plateclock','--image',str(source),'--out',str(tmp_path/'demo'),'--no-audio'])
    main()
    assert json.loads((tmp_path/'demo.json').read_text())['status']=='ok'
    assert (tmp_path/'demo.txt').is_file() and (tmp_path/'demo_overlay.png').is_file()

@pytest.mark.parametrize('angle',range(0,360,18))
def test_twenty_synthetic_layouts(angle,cfg):
    import cv2
    image=np.full((321,321,3),25,np.uint8);cv2.circle(image,(160,160),125,(235,235,235),-1)
    class Model:
        def predict(self,image):
            labels=np.zeros(image.shape[:2],np.uint8)
            t=np.deg2rad(angle)
            cv2.circle(labels,(round(160+65*np.sin(t)),round(160-65*np.cos(t))),22,66,-1)
            return labels,np.ones(labels.shape,np.float32)
    result=run(image,Model(),cfg)
    assert result['status']=='ok' and not lint(result['text'])
    assert result['items'][0].hour==__import__('plateclock.plate.geometry',fromlist=['hour_at']).hour_at(angle)

def test_ten_written_examples():
    from pathlib import Path
    examples=Path('report/style_examples.txt').read_text().splitlines()
    assert len(examples)==10
    assert all(not lint(text) for text in examples)

def test_evaluation_workflow_synthetic(monkeypatch,tmp_path):
    import csv,json,sys,types,cv2,importlib.util
    from pathlib import Path
    from PIL import Image
    image=np.full((321,321,3),25,np.uint8);cv2.circle(image,(160,160),125,(235,235,235),-1)
    labels=np.zeros((321,321),np.uint8);cv2.circle(labels,(160,100),22,66,-1)
    Image.fromarray(image).save(tmp_path/'image.png');Image.fromarray(labels).save(tmp_path/'mask.png')
    (tmp_path/'classes.json').write_text(json.dumps(CLASS_NAMES))
    with (tmp_path/'subset.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','image_path','mask_path'])
        w.writerows((str(i),'image.png','mask.png') for i in range(300))
    class Model:
        def __init__(self,*args):pass
        def predict(self,image):return labels,np.full(labels.shape,.95,np.float32)
    monkeypatch.setitem(sys.modules,'plateclock.plate.segment_food',types.SimpleNamespace(FoodSegmenter=Model))
    monkeypatch.setattr(sys,'argv',['evaluate','--root',str(tmp_path),'--subset',str(tmp_path/'subset.csv'),'--out',str(tmp_path/'out')])
    spec=importlib.util.spec_from_file_location('week2_eval',Path('scripts/evaluate_week2.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
    output=tmp_path/'out'
    summary=json.loads((output/'summary.json').read_text())
    assert summary['images']==300 and summary['calibration_candidates']==150
    assert (output/'confidence_curves.png').is_file()
    holdout=json.loads((output/'holdout_metrics.json').read_text())
    assert holdout['omit_below']['candidates']==150
    assert holdout['omit_below']['precision']==1.0
    rows=list(csv.DictReader((output/'candidates.csv').open()))
    assert all(r['correct']=='1' for r in rows)
    split=json.loads((output/'split.json').read_text())
    assert sum(v=='holdout' for v in split.values())==150

