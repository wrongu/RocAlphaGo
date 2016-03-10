from AlphaGo.models.game_converter import game_converter

test_features = ["board", "turns_since", "liberties", "capture_size",
	"self_atari_size", "liberties_after", "sensibleness", "zeros"]

gc = game_converter()
for s_a_tuple in gc.convert_game('tests/test_sgfs/AlphaGo-vs-Lee-Sedol-20160310-first10only.sgf', test_features):
     pass
     
