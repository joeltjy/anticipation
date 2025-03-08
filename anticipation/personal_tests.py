from convert import *
from ops import *
import tokenize

import unittest

midifile = "../../lmd_full/0/0a0a2b0e4d3b7bf4c5383ba025c4683e.mid"
# interarrival_output = midi_to_interarrival(midifile)
# print("Interarrival", interarrival_output[:100])

# compound_output = midi_to_compound(midifile)
# print("Compound", compound_output[:100])

# events_output = compound_to_events(compound_output)
# print("Events", events_output[:100])

def convert_midi_test(filename, debug=False):
    try:
        tokens_compound = midi_to_compound(filename, debug=debug)
        tokens_without_velocity = midi_to_events(filename, debug=debug, include_velocity = False)
        tokens_with_velocity = midi_to_events(filename, debug=debug, include_velocity = True)
        

    except Exception:
        if debug:
            print('Failed to process: ', filename)

        return 1

    
    # with open(f"{filename}.compound.txt", 'w') as f:
    #     f.write(' '.join(str(tok) for tok in tokens))

    print("done")
    return (tokens_compound, tokens_without_velocity, tokens_with_velocity)


class TestChanges(unittest.TestCase):

    def setUp(self):
        (self.tokens_compound, self.tokens_without_velocity, self.tokens_with_velocity) = convert_midi_test(midifile)

    def test_compound_to_events(self):
        #print(self.tokens_compound[:100])
        #print(self.tokens_with_velocity[:80])
        #print(self.tokens_without_velocity[:60])
        assert self.tokens_compound[90] == self.tokens_with_velocity[72] == self.tokens_without_velocity[54]

    def test_events_to_compound(self):
        generated_tokens_compound = events_to_compound(self.tokens_with_velocity, include_velocity=True)
        n = len(generated_tokens_compound)
        for i in range(n):
            assert generated_tokens_compound[i] == self.tokens_compound[i], "disagrees at position i"
        
        generated_tokens_compound_2 = events_to_compound(self.tokens_without_velocity)
        n = len(generated_tokens_compound_2)
        for i in range(n):
            if i % 5 < 4:
                assert generated_tokens_compound_2[i] == self.tokens_compound[i], "disagrees at position i"

    def test_print_tokens(self):
        # visual check
        #print_tokens(self.tokens_without_velocity, False)
        #print_tokens(self.tokens_with_velocity, True)
        assert True
    
    def test_clip(self):
        t1 = clip(self.tokens_without_velocity, 0, 1)
        t2 = clip(self.tokens_with_velocity, 0, 1, include_velocity=True)
        assert t1[-3] == t2[-4]
        assert len(t1) * 4  == len(t2) * 3

    def test_mask(self):
        t1 = mask(self.tokens_without_velocity, 0, 5)
        t2 = mask(self.tokens_with_velocity, 0, 5, include_velocity=True)
        assert t1[-3] == t2[-4]
        assert len(t1) * 4  == len(t2) * 3

    def test_anticipate(self):
        t1 = mask(self.tokens_without_velocity, 0, 5)
        t2 = mask(self.tokens_with_velocity, 0, 5, include_velocity=True)
        t1e = []
        t1c = []
        t2e = []
        t2c = []
        for i, tok in enumerate(t1):
            if i % 6 < 3:
                t1e.append(tok)
            else:
                t1c.append(tok)
        for i, tok in enumerate(t2):
            if i % 8 < 4:
                t2e.append(tok)
            else:
                t2c.append(tok)
        t1a, _ = anticipate(t1e, t1c)
        t2a, _ = anticipate(t2e, t2c, include_velocity=True)
        for i in range(len(t1a)//4):
            assert t1a[i*3] == t2a[i*4]
    
    def test_min_time(self):
        t1 = min_time(self.tokens_without_velocity)
        t2 = min_time(self.tokens_with_velocity, include_velocity=True)
        assert t1 == t2
    
    def test_max_time(self):
        t1 = max_time(self.tokens_without_velocity)
        t2 = max_time(self.tokens_with_velocity, include_velocity=True)
        assert t1 == t2
    
    def test_get_instruments(self):
        t1 = get_instruments(self.tokens_without_velocity)
        t2 = get_instruments(self.tokens_with_velocity, include_velocity=True)
        print(t1)
        print(t2)
        for i in range(len(t1)):
            assert t1[i] == t2[i]

    def test_midi_to_interarrival(self):
        interarrival_output = midi_to_interarrival(midifile)
        #print("Interarrival", interarrival_output[:100])
        interarrival_output_with_velocity = midi_to_interarrival(midifile, include_velocity=True)
        #print("Interarrival with velocity", interarrival_output_with_velocity[:100])
        midi = interarrival_to_midi(interarrival_output_with_velocity, include_velocity=True)
        #print(midi.tracks[0][:100])
        actual_midi = mido.MidiFile(midifile)
        #print(actual_midi.tracks[2][:100])
        assert True
    
    def test_tokenize(self):
        file = "../maestrodata/2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_02_Track02_wav.midi.compound.txt"
        tokenize.tokenize([file], "tokenized.txt", 1, 0, False, include_velocity=True)
        assert True
        

if __name__ == "__main__":
    unittest.main()
