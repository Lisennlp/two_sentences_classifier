import sys

before_subj_symbols = ['，', '。', '？', '！', '：', '；', '……', '…', '——', '—', '~', '～', '-', '－']
before_subj_symbols += [',', '.', '?', '!', ':', ';']


def get_speaker(line):
    assert '::' in line, line
    return line.split('::')[0]


def get_speech(line):
    assert '::' in line, line
    return line.split('::', maxsplit=1)[1]


def is_cjk_char(char):
    return '\u4e00' <= char <= '\u9fff'


def get_span(lines, center, sep=None, radius=2):
    seps = ['…', '—', '.', '·', '-', '。'] if sep is None else [sep]
    speaker, speech = lines[center].split('::', maxsplit=1)
    assert speaker == '旁白', speaker
    # assert '本章完' not in speech, speech
    indexes = [center]
    n = 0
    for i in reversed(range(0, center)):
        if is_chapter_name(lines[i]):
            continue
        speech = get_speech(lines[i])
        if not all(c in seps for c in speech):    # and '本章完' not in speech:
            indexes.insert(0, i)
            n += 1
        if n == radius:
            break
    n = 0
    for i in range(center + 1, len(lines)):
        if is_chapter_name(lines[i]): continue
        speech = get_speech(lines[i])
        if not all(c in seps for c in speech):    # and '本章完' not in speech:
            indexes.append(i)
            n += 1
        if n == radius:
            break
    return indexes if len(indexes) == radius * 2 + 1 else None


def normalize(line):
    line = line.replace('::', '：').replace('\n', '')    #.replace('旁白：', '')
    if line[-1] not in before_subj_symbols:
        line = line + '。'
    return line


def dump_span(span, f=sys.stdout):
    title, sep, sep_count, lines, label = span
    lines_str = '||'.join([normalize(line) for line in lines])
    label = str(int(label))    # bool -> str
    print('\t'.join([lines_str, label, title, sep]), file=f)


def is_chapter_name(line):
    b = '::' not in line and line.split(
    )[0][0] == '第' and line.split()[0][-1] == '话'    # '第23话' or '第45话 回家'
    # if line.startswith('第6话'): assert b, line
    return b


def filter_lines(lines, keep_chapter_name=False):
    ret_lines = []
    for line in lines:
        if '::' not in line:
            assert is_chapter_name(line), line
            if keep_chapter_name:
                ret_lines.append(line)
            continue   
        speaker, speech = line.split('::', maxsplit=1)
        if speaker.startswith('旁白'):
            if any(s in speech for s in ['正文', '本章完', '待续', '未完', '分割线', '卡文']) or \
                all(not is_cjk_char(c) or c == '卡' for c in speech) and any(c == '卡' for c in speech):
                continue
            if speech.strip() == '':
                continue
        ret_lines.append(line)
    return ret_lines
