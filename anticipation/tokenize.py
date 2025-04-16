"""
Top-level functions for preprocessing data to be used for training.
"""

from tqdm import tqdm

import numpy as np

from anticipation import ops
from anticipation.config import *
from anticipation.vocab import *
from anticipation.convert import compound_to_events, midi_to_interarrival

# all changes done

# changed
def extract_spans(all_events, rate, include_velocity=False):
    events = []
    controls = []
    span = True
    next_span = end_span = TIME_OFFSET+0

    if include_velocity:
        for time, dur, note, vel in zip(all_events[0::4],all_events[1::4],all_events[2::4], all_events[3::4]):
            assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

            # end of an anticipated span; decide when to do it again (next_span)
            if span and time >= end_span:
                span = False
                next_span = time+int(TIME_RESOLUTION*np.random.exponential(1./rate))

            # anticipate a 3-second span
            if (not span) and time >= next_span:
                span = True
                end_span = time + DELTA*TIME_RESOLUTION

            if span:
                # mark this event as a control
                controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note, CONTROL_OFFSET_VELOCITY + vel])
            else:
                events.extend([time, dur, note, vel])

        return events, controls
    else:   
        for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
            assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

            # end of an anticipated span; decide when to do it again (next_span)
            if span and time >= end_span:
                span = False
                next_span = time+int(TIME_RESOLUTION*np.random.exponential(1./rate))

            # anticipate a 3-second span
            if (not span) and time >= next_span:
                span = True
                end_span = time + DELTA*TIME_RESOLUTION

            if span:
                # mark this event as a control
                controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
            else:
                events.extend([time, dur, note])

        return events, controls

# changed
ANTICIPATION_RATES = 10
def extract_random(all_events, rate, include_velocity=False):
    events = []
    controls = []
    if include_velocity:
        for time, dur, note, vel in zip(all_events[0::4],all_events[1::4],all_events[2::4], all_events[3::4]):
            assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

            if np.random.random() < rate/float(ANTICIPATION_RATES):
                # mark this event as a control
                controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note, CONTROL_OFFSET_VELOCITY + vel])
            else:
                events.extend([time, dur, note, vel])

        return events, controls
    
    else:
        for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
            assert(note not in [SEPARATOR, REST]) # shouldn't be in the sequence yet

            if np.random.random() < rate/float(ANTICIPATION_RATES):
                # mark this event as a control
                controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
            else:
                events.extend([time, dur, note])

        return events, controls


def pitch_augmentation(all_events, pitch_add=None, include_velocity=False):
    events = []
    
    if pitch_add is None:
        pitch_add = np.random.randint(-12, 12)
    
    if include_velocity:
        for time, dur, note, vel in zip(all_events[0::4],all_events[1::4],all_events[2::4], all_events[3::4]):
            assert note not in [SEPARATOR, REST] # shouldn't be in the sequence yet
            assert note < CONTROL_OFFSET
        
            instr = (note-NOTE_OFFSET)//2**7
            pitch = (note-NOTE_OFFSET)%2**7

            pitch += pitch_add
            if pitch < 0:
                pitch = 0
            elif pitch > 127:
                pitch = 127

            note = instr*2**7 + pitch

            events.extend([time, dur, note, vel])
    else:
        for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
            assert note not in [SEPARATOR, REST] # shouldn't be in the sequence yet
            assert note < CONTROL_OFFSET

            instr = (note-NOTE_OFFSET)//2**7
            pitch = (note-NOTE_OFFSET)%2**7

            pitch += pitch_add
            if pitch < 0:
                pitch = 0
            elif pitch > 127:
                pitch = 127

            note = instr*2**7 + pitch

            events.extend([time, dur, note])

    return events

def tempo_augmentation(all_events, tempo_factor=None, include_velocity=False):
    events = []

    if tempo_factor is None:
        tempo_factor = np.random.uniform(0.5, 2.0)
        
    if include_velocity:
        for time, dur, note, vel in zip(all_events[0::4],all_events[1::4],all_events[2::4], all_events[3::4]):
            assert note not in [SEPARATOR, REST] # shouldn't be in the sequence yet
            assert note < CONTROL_OFFSET

            time = int(time * tempo_factor)
            dur = min(DUR_OFFSET + int((dur-DUR_OFFSET) * tempo_factor), NOTE_OFFSET-1)
            
            if time <= MAX_TIME:
                events.extend([time, dur, note, vel])
    else:
        for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
            assert note not in [SEPARATOR, REST] # shouldn't be in the sequence yet
            assert note < CONTROL_OFFSET

            time = int(time * tempo_factor)
            dur = min(DUR_OFFSET + int((dur-DUR_OFFSET) * tempo_factor), NOTE_OFFSET-1)
            
            if time <= MAX_TIME:
                events.extend([time, dur, note])

    return events

def velocity_augmentation(all_events, velocity_factor=None):
    events = []

    if velocity_factor is None:
        velocity_factor = np.random.uniform(0.5, 2.0)
        
    for time, dur, note, vel in zip(all_events[0::4],all_events[1::4],all_events[2::4], all_events[3::4]):
        assert note not in [SEPARATOR, REST] # shouldn't be in the sequence yet
        assert note < CONTROL_OFFSET

        vel = vel - VELOCITY_OFFSET
        vel = int(vel * velocity_factor)
        vel = vel + VELOCITY_OFFSET

        if vel < 0:
            vel = 0
        elif vel > 127:
            vel = 127

        events.extend([time, dur, note, vel])

    return events
# changed
def extract_instruments(all_events, instruments, include_velocity=False):
    events = []
    controls = []

    if include_velocity:
        for time, dur, note, vel in zip(all_events[0::4],all_events[1::4],all_events[2::4], all_events[3::4]):
            assert note < CONTROL_OFFSET         # shouldn't be in the sequence yet
            assert note not in [SEPARATOR, REST] # these shouldn't either

            instr = (note-NOTE_OFFSET)//2**7
            if instr in instruments:
                # mark this event as a control
                controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note, CONTROL_OFFSET_VELOCITY + vel])
            else:
                events.extend([time, dur, note, vel])

        return events, controls
    else:
        for time, dur, note in zip(all_events[0::3],all_events[1::3],all_events[2::3]):
            assert note < CONTROL_OFFSET         # shouldn't be in the sequence yet
            assert note not in [SEPARATOR, REST] # these shouldn't either

            instr = (note-NOTE_OFFSET)//2**7
            if instr in instruments:
                # mark this event as a control
                controls.extend([CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+note])
            else:
                events.extend([time, dur, note])

        return events, controls

# changed
def maybe_tokenize(compound_tokens, include_velocity = False):
    # skip sequences with very few events
    if len(compound_tokens) < COMPOUND_SIZE*MIN_TRACK_EVENTS:
        return None, None, 1 # short track

    events, truncations = compound_to_events(compound_tokens, stats=True, include_velocity=include_velocity)
    end_time = ops.max_time(events, seconds=False, include_velocity=include_velocity)

    # don't want to deal with extremely short tracks
    if end_time < TIME_RESOLUTION*MIN_TRACK_TIME_IN_SECONDS:
        return None, None, 1 # short track

    # don't want to deal with extremely long tracks
    if end_time > TIME_RESOLUTION*MAX_TRACK_TIME_IN_SECONDS:
        return None, None, 2 # long track

    # skip sequences more instruments than MIDI channels (16)
    if len(ops.get_instruments(events, include_velocity=include_velocity)) > MAX_TRACK_INSTR:
        return None, None, 3 # too many instruments

    return events, truncations, 0

# changed
def tokenize_ia(datafiles, output, augment_factor, idx=0, debug=False, include_velocity = False):
    assert augment_factor == 1 # can't augment interarrival-tokenized data

    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    with open(output, 'w') as outfile:
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            with open(filename, 'r') as f:
                _, _, status = maybe_tokenize([int(token) for token in f.read().split()], include_velocity)

            if status > 0:
                stats[status-1] += 1
                continue

            filename = filename[:-len('.compound.txt')] # get the original MIDI

            # already parsed; shouldn't raise an exception
            tokens, truncations = midi_to_interarrival(filename, stats=True, include_velocity=include_velocity)
            tokens[0:0] = [MIDI_SEPARATOR]
            concatenated_tokens.extend(tokens)
            all_truncations += truncations

            # write out full sequences to file
            while len(concatenated_tokens) >= CONTEXT_SIZE:
                seq = concatenated_tokens[0:CONTEXT_SIZE]
                concatenated_tokens = concatenated_tokens[CONTEXT_SIZE:]
                outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                seqcount += 1

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)

# changed
def tokenize(datafiles, output, augment_factor, idx=0, debug=False, include_velocity = False):
    print("in tokenize function")
    tokens = []
    all_truncations = 0
    seqcount = rest_count = 0
    stats = 4*[0] # (short, long, too many instruments, inexpressible)
    np.random.seed(0)

    tokens_per_event = 3
    if include_velocity:
        tokens_per_event = 4

    with open(output, 'w') as outfile:
        
        concatenated_tokens = []
        for j, filename in tqdm(list(enumerate(datafiles)), desc=f'#{idx}', position=idx+1, leave=True):
            
            with open(filename, 'r') as f:
                print("file found", filename)
                all_events, truncations, status = maybe_tokenize([int(token) for token in f.read().split()], include_velocity=include_velocity)
                print("maybe tokenized")

                print("all events", all_events)
            if status > 0:
                stats[status-1] += 1
                continue

            instruments = list(ops.get_instruments(all_events,include_velocity=include_velocity).keys())
            end_time = ops.max_time(all_events, seconds=False, include_velocity=include_velocity)

            print("instruments", instruments)
            print("end time", end_time)
            # different random augmentations
            for k in range(augment_factor):
                if k % 10 == 0:
                    # no augmentation
                    events = all_events.copy()
                    controls = []
                elif k % 10 == 1:
                    # span augmentation
                    lmbda = .05
                    events, controls = extract_spans(all_events, lmbda, include_velocity=include_velocity)
                elif k % 10 == 2:
                    # velocity augmentation
                    events = velocity_augmentation(all_events, velocity_factor=None)
                    controls = []
                elif k % 10 == 3:
                    # tempo augmentation
                    events = tempo_augmentation(all_events, tempo_factor=None, include_velocity=include_velocity)
                    controls = []
                elif k % 10 == 4:
                    # pitch augmentation
                    events = pitch_augmentation(all_events, pitch_add=None, include_velocity=include_velocity)
                    controls = []
                elif k % 10 < 7:
                    # random augmentation
                    r = np.random.randint(1,ANTICIPATION_RATES)
                    events, controls = extract_random(all_events, r, include_velocity=include_velocity)
                else:
                    if len(instruments) > 1:
                        # instrument augmentation: at least one, but not all instruments
                        u = 1+np.random.randint(len(instruments)-1)
                        subset = np.random.choice(instruments, u, replace=False)
                        events, controls = extract_instruments(all_events, subset, include_velocity=include_velocity)
                    else:
                        # no augmentation
                        events = all_events.copy()
                        controls = []

                if len(concatenated_tokens) == 0:
                    z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

               
                all_truncations += truncations
                events = ops.pad(events, end_time, include_velocity=include_velocity)
                rest_count += sum(1 if tok == REST else 0 for tok in events[2::3])
                tokens, controls = ops.anticipate(events, controls, include_velocity=include_velocity)
                assert len(controls) == 0 # should have consumed all controls (because of padding)
                
                if include_velocity:
                    tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR, SEPARATOR]
                else:
                    tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
                concatenated_tokens.extend(tokens)

                # write out full sequences to file
                max_length = (CONTEXT_SIZE-1) - (CONTEXT_SIZE-1) % tokens_per_event
                while len(concatenated_tokens) >= max_length:
                    seq = concatenated_tokens[0:max_length]
                    concatenated_tokens = concatenated_tokens[max_length:]

                    print("min time", ops.min_time(seq, seconds=False, include_velocity=include_velocity))
                    # relativize time to the context
                    seq = ops.translate(seq, -ops.min_time(seq, seconds=False, include_velocity=include_velocity), seconds=False, include_velocity=include_velocity)
                    print("translated")
                    assert ops.min_time(seq, seconds=False, include_velocity=include_velocity) == 0
                    if ops.max_time(seq, seconds=False, include_velocity=include_velocity) >= MAX_TIME:
                        stats[3] += 1
                        continue

                    # if seq contains SEPARATOR, global controls describe the first sequence
                    seq.insert(0, z)

                    outfile.write(' '.join([str(tok) for tok in seq]) + '\n')
                    seqcount += 1

                    # grab the current augmentation controls if we didn't already
                    z = ANTICIPATE if k % 10 != 0 else AUTOREGRESS

    if debug:
        fmt = 'Processed {} sequences (discarded {} tracks, discarded {} seqs, added {} rest tokens)'
        print(fmt.format(seqcount, stats[0]+stats[1]+stats[2], stats[3], rest_count))

    return (seqcount, rest_count, stats[0], stats[1], stats[2], stats[3], all_truncations)
