"""Strict plate grammar: count, ordering, vocabulary, hedges and length."""
import re
from plateclock.language.style_guide import load_style

def lint(text, mode='full', style=None):
    s = style or load_style()
    errors = []
    sentences = [part.strip() for part in text.split('.') if part.strip()]
    if not text.endswith('.') or '..' in text or any(c in text for c in '!?;\n'):
        errors.append('Use period-terminated sentences only')
    if not sentences or len(sentences) > s['max_sentences']:
        errors.append('Sentence limit')
    if any(len(part.split()) > s['max_words'] for part in sentences):
        errors.append('Word limit')
    if re.search(r'\b(?:' + '|'.join(map(re.escape, s['banned'])) + r')\b', text, re.I):
        errors.append('Banned word')
    if text in (s['failure'], s['empty']):
        return errors
    labels = '(?:' + '|'.join(map(re.escape, s['labels'])) + ')'
    zones = '(?:' + '|'.join(map(re.escape, s['zones'])) + ')'
    hedge = '(?:' + re.escape(s['hedge']) + ' )?'
    if mode == 'summary':
        shares = '(?:' + '|'.join(map(re.escape, [s['share_large'], s['share_small'], s['share_medium']])) + ')'
        if not re.fullmatch(hedge + shares + ' ' + labels + r'\.', text, re.I):
            errors.append('Invalid summary grammar')
        return errors
    if mode != 'full':
        return errors + ['Unknown mode']
    opening = re.fullmatch(r'([1-5]) (item|items) identified', sentences[0]) if sentences else None
    if not opening or int(opening[1]) != len(sentences)-1 or opening[2] != ('item' if opening[1]=='1' else 'items'):
        errors.append('Opening count mismatch')
    if len(sentences)-1 > s['max_items']:
        errors.append('Item limit')
    hours = []
    pattern = hedge + labels + r" at (1[0-2]|[1-9]) o'clock, " + zones
    for part in sentences[1:]:
        m = re.fullmatch(pattern, part)
        if not m:
            errors.append('Invalid item grammar: ' + part)
        else:
            hours.append(int(m[1]) % 12)
    if hours != sorted(hours):
        errors.append('Items must be clockwise from 12')
    return errors

def require_valid(text, mode='full', style=None):
    errors = lint(text, mode, style)
    if errors:
        raise ValueError('Style lint failed: ' + '; '.join(errors))
