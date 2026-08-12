"""Prompt bank for teacher data generation.

Composition is deliberate rather than arbitrary. The deployment story is a
persistent world W that a stream of events continues, so the bank is weighted
towards content with a stable subject and continuous, non-cut motion -- that is
the distribution the student has to stay in over a long rollout. Hard cuts,
crowds and rapid scene changes are exactly what a block-causal model with a
sliding K/V window cannot represent, so they are kept out.

Roughly half is human subjects (the papers' interaction use case), the rest
animals, nature and urban scenes so the model does not collapse onto one mode.
"""

PEOPLE = [
    "A woman with long dark hair speaking directly to the camera, natural facial expressions, soft studio lighting, shallow depth of field",
    "A man in a grey sweater talking to the camera and gesturing calmly, warm indoor lighting, blurred bookshelf background",
    "A young woman smiling and nodding while listening, soft window light from the left, neutral background",
    "An older man with a short white beard telling a story to the camera, warm lamp light, cozy living room",
    "A woman in a yellow raincoat looking around, light rain falling, overcast daylight, city street behind her",
    "A chef in a white coat carefully plating a dish, overhead kitchen lighting, steam rising",
    "A barista steaming milk behind a counter, morning light through a large window, cafe interior",
    "A violinist playing on a small stage, warm spotlight, dark background",
    "A woman reading a book by a window, turning a page, soft afternoon light, dust motes in the air",
    "A man drinking coffee from a mug and looking out of a rainy window, muted daylight",
    "A painter working on a canvas, brush strokes visible, studio skylight, paint-spattered apron",
    "A woman practicing yoga on a mat, slow controlled movement, morning light on a wooden floor",
    "A doctor in a white coat speaking to the camera in a bright clinic, professional lighting",
    "A teacher writing on a whiteboard and turning to speak, classroom, fluorescent light",
    "A man repairing a bicycle wheel in a garage, focused expression, work lamp lighting",
    "A woman arranging flowers in a vase, bright kitchen, sunlight on the counter",
    "A street musician playing guitar on a sidewalk, passersby blurred behind, golden hour",
    "A potter shaping clay on a spinning wheel, hands wet with slip, warm workshop light",
    "A woman in a wool coat walking slowly through a park in autumn, leaves falling",
    "A man in a suit adjusting his tie in front of a mirror, hotel room, warm lamps",
    "A child blowing bubbles in a garden, sunlight through the bubbles, summer afternoon",
    "A woman laughing while talking on the phone, cafe interior, bokeh lights behind",
    "A dancer moving slowly in an empty studio, mirrors and barre, cool daylight",
    "A fisherman mending a net on a dock, weathered hands, overcast coastal light",
    "A woman scientist looking into a microscope and then up at the camera, laboratory lighting",
    "A man playing chess alone, moving a piece, window light across the board",
    "A woman with curly hair singing into a studio microphone, headphones on, dim red light",
    "An elderly woman knitting in an armchair, fireplace glow, quiet living room",
    "A carpenter sanding a wooden plank, sawdust in the air, workshop window light",
    "A woman tying her running shoes on a park bench, early morning, mist in the background",
    "A man walking a dog along a canal, evening light, calm water",
    "A woman applying makeup in front of a lit mirror, dressing room, soft bulbs",
    "A librarian pulling a book from a high shelf, tall wooden stacks, warm reading lamps",
    "A woman in a lab coat writing notes on a clipboard, bright modern interior",
    "A man kneading dough on a floured counter, bakery, morning sun through a window",
    "A woman in a denim jacket leaning on a railing overlooking a city, wind in her hair",
    "A surfer in a wetsuit walking up a beach carrying a board, late afternoon sun",
    "A woman gardening, planting a seedling into dark soil, bright overcast daylight",
    "A man playing piano in a dim room, hands on the keys, single warm lamp",
    "A woman photographer adjusting a camera lens, city rooftop, dusk",
]

ANIMALS = [
    "A ginger cat sitting on a windowsill, tail flicking, sunlight across its fur",
    "A golden retriever running across a grassy field toward the camera, sunny day",
    "A horse grazing in a misty meadow at dawn, breath visible in the cold air",
    "A red fox stepping carefully through fresh snow in a pine forest",
    "An owl slowly turning its head on a branch, dappled forest light",
    "A pod of dolphins swimming just under a clear turquoise surface",
    "A hummingbird hovering at a red flower, wings blurred, bright garden",
    "A sheepdog herding sheep across a green hillside, overcast light",
    "A tabby kitten batting at a dangling string, wooden floor, warm indoor light",
    "A deer lifting its head alertly in a clearing, early morning fog",
    "A parrot with green and red feathers preening on a branch, tropical foliage",
    "A school of small silver fish turning together over a coral reef, shafts of sunlight",
    "A bumblebee moving across lavender flowers, summer sunlight, shallow focus",
    "An elephant slowly walking across a dry savannah, dust rising, late afternoon",
    "A black cat stretching on a sunlit rug, slow movement, quiet room",
    "A hawk perched on a fence post scanning the field, wind in the grass",
]

NATURE = [
    "Waves rolling onto a rocky shore at sunset, spray catching the light",
    "A mountain stream running over smooth stones, sunlight through overhanging leaves",
    "Tall grass moving in the wind on a hillside, clouds drifting overhead",
    "Snow falling slowly through a dense pine forest, quiet grey light",
    "A waterfall in a green canyon, mist rising, shafts of sunlight",
    "Autumn leaves drifting down onto a still pond, reflections rippling",
    "A field of sunflowers swaying, bright blue sky with scattered clouds",
    "Clouds moving over a desert mesa, long shadows, late afternoon",
    "A lavender field in bloom, gentle wind, hazy summer sun",
    "Fog rolling slowly through a valley of dark evergreens at dawn",
    "Rain falling on the surface of a lake, concentric ripples, grey daylight",
    "A campfire burning at dusk, sparks rising, forest silhouettes behind",
    "Northern lights shifting slowly above a snowy ridge, deep blue night",
    "A wheat field rippling in the wind under a dramatic evening sky",
    "Palm fronds moving in a warm breeze against a bright tropical sky",
    "Ice floating slowly down a wide grey river, bare winter trees on the bank",
]

URBAN = [
    "A quiet city street at night, wet asphalt reflecting neon signs, light rain",
    "Steam rising from a manhole on a cold morning, cars passing slowly",
    "A cafe terrace in the afternoon, awning moving slightly, people seated",
    "A subway train arriving at a platform, motion blur, fluorescent light",
    "Traffic crossing a bridge at dusk, headlights and taillights, city skyline behind",
    "A narrow European alley with laundry hanging, warm afternoon light",
    "Rain on a bus window, blurred city lights beyond the glass",
    "A market stall with fresh produce, vendor arranging fruit, bright daylight",
    "A quiet bookshop interior, dust in a shaft of sunlight, shelves of books",
    "A rooftop at golden hour overlooking a dense city, antennas and water tanks",
    "An empty basketball court at dusk, net moving slightly in the wind",
    "A ferry crossing a harbour, gulls following, overcast light",
    "A neon-lit ramen shop at night seen from the street, steam in the doorway",
    "An old tram turning a corner on cobblestones, autumn trees along the street",
]

OBJECTS = [
    "Coffee being poured into a glass cup, swirling crema, morning kitchen light",
    "A candle flame flickering in a dark room, wax slowly melting",
    "Ink diffusing into a glass of clear water, soft studio lighting",
    "A record spinning on a turntable, needle in the groove, warm lamp light",
    "Dough rising in a bowl in time lapse, kitchen window light",
    "Water droplets running down a cold glass bottle, dark background",
    "A mechanical watch movement ticking, extreme close-up, jeweller's lighting",
    "Paint colours swirling together on a wet canvas, overhead studio light",
    "Steam curling from a bowl of soup on a wooden table, warm evening light",
    "A kite flying against a blue sky, string taut, wind moving the tail",
]

ALL = PEOPLE + ANIMALS + NATURE + URBAN + OBJECTS


from .prompts_ext import EXT_ALL, REPLACEMENTS, assert_no_minors  # noqa: E402


def prompt_bank():
    bank = list(ALL)
    for i, text in REPLACEMENTS.items():
        bank[i] = text
    bank += EXT_ALL
    assert_no_minors(bank)
    return bank


def categories():
    from . import prompts_ext as _x
    return {'people': PEOPLE + _x.PEOPLE_EXT, 'animals': ANIMALS + _x.ANIMALS_EXT,
            'nature': NATURE + _x.NATURE_EXT, 'urban': URBAN + _x.URBAN_EXT,
            'objects': OBJECTS + _x.OBJECTS_EXT}


if __name__ == '__main__':
    for name, lst in categories().items():
        print(f'{name:8s} {len(lst)}')
    print(f'{"TOTAL":8s} {len(prompt_bank())}  (v5 bank {len(ALL)} at indices 0-95)')
