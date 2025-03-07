from convert import *
from ops import *

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
        print(self.tokens_compound[:100])
        print(self.tokens_with_velocity[:80])
        print(self.tokens_without_velocity[:60])
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
        print_tokens(self.tokens_without_velocity, False)
        print_tokens(self.tokens_with_velocity, True)
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


if __name__ == "__main__":
    unittest.main()
