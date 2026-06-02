import json

from datasets import load_dataset

def load_generic_dataset() -> list[str]:
    more_sentences = [
    # --- Short & Functional ---
    "Run the script.",
    "Access denied.",
    "Sunlight fades.",
    "Keep it simple.",
    "Logic dictates peace.",
    "Water the plants.",
    "Gravity is constant.",
    "The clock ticked.",
    "System online.",
    "Believe in yourself.",

    # --- Daily Life & Casual ---
    "I'm pretty sure my laundry room is a portal to another dimension where only left socks are allowed to exist.",
    "If you see me talking to myself, just move along; I’m having a parent-teacher conference with my inner child.",
    "There is nothing quite like the smell of fresh rain on hot asphalt during a humid July afternoon in the city.",
    "I tried to make sourdough bread from scratch, but I ended up with a brick that could arguably be used for home defense.",
    "Does anyone else feel like they need a nap immediately after waking up from an eight-hour sleep cycle?",
    "The grocery store was out of avocados, which effectively ruined my plans for a fancy Sunday brunch.",
    "My cat spends most of his day staring at a blank wall, making me wonder if he sees ghosts or is just deeply bored.",
    "I missed the bus by thirty seconds and had to wait in the freezing cold for the next one, which was naturally late.",
    "Is it still considered a 'hobby' if I only do it once every six months when I feel guilty about spending money on it?",
    "The internet went out for ten minutes, and I realized I have absolutely no idea how to survive in the physical world.",

    # --- Professional & Corporate ---
    "Please find the attached document regarding the proposed budget cuts for the upcoming fiscal year.",
    "We need to pivot our strategy to better align with the shifting paradigms of the global digital marketplace.",
    "The synergy between the marketing and product teams has resulted in a 15% increase in user retention rates.",
    "I'll loop you in on the email chain so you can provide your insights on the architectural constraints of the project.",
    "Let’s take this offline and circle back once we have more granular data from the third-quarter reports.",
    "The executive summary highlights the necessity of a robust infrastructure to support our anticipated hyper-growth phase.",
    "Our core competency lies in delivering bespoke solutions that address the unique pain points of our diverse clientele.",
    "The stakeholder meeting was adjourned early due to a lack of consensus on the proposed mitigation strategies.",
    "Efficiency is not just about speed; it is about optimizing resource allocation to minimize waste and maximize output.",
    "We are currently in a holding pattern while we wait for legal to sign off on the nondisclosure agreements.",

    # --- Scientific & Technical ---
    "The mitochondria are often described as the powerhouse of the cell because they generate most of the chemical energy.",
    "Entropy is a measure of the disorder or randomness within a closed system, according to the second law of thermodynamics.",
    "The event horizon of a black hole marks the point of no return, where even light cannot escape the gravitational pull.",
    "Neural networks are designed to mimic the human brain's structure, allowing machines to learn from vast amounts of data.",
    "Carbon sequestration involves capturing atmospheric carbon dioxide to mitigate the effects of global climate change.",
    "The tectonic plates are constantly moving at a rate of a few centimeters per year, driven by mantle convection.",
    "An asymptotic analysis provides a way to describe the limiting behavior of a function as the input grows toward infinity.",
    "The Heisenberg Uncertainty Principle states that we cannot simultaneously know the exact position and momentum of a particle.",
    "Bioluminescence is the production and emission of light by a living organism as the result of a chemical reaction.",
    "The integration of blockchain technology could potentially revolutionize the transparency of global supply chain management.",

    # --- Literary & Poetic (The "Very Long" Category) ---
    "The sea was a restless beast that night, clawing at the jagged cliffs with foamy white talons while the lighthouse beam cut through the salt-spray like a desperate, searching eye looking for a lost lover in the dark.",
    "He wandered through the dusty corridors of the abandoned library, feeling the weight of thousands of unread stories pressing against him, each volume a silent monument to a mind that had long since crumbled into the earth.",
    "Under the canopy of the ancient willow, where the silver leaves danced in the moonlight, she whispered a secret that had been held tight in her heart for twenty years, letting the words drift away on the breeze like dandelion seeds.",
    "The city at midnight is a tapestry of neon lights and long shadows, where the hum of electricity replaces the song of birds and the pavement holds the warmth of a sun that has long since vanished beneath the iron horizon.",
    "Looking back at the path he had traveled, he realized that the detours he once considered failures were actually the very moments that shaped his character and led him to the quiet clearing where he finally found peace.",
    "The protagonist stood at the edge of the crater, staring into the molten heart of the volcano and wondering if the artifact he carried was worth the price of the world he was about to leave behind forever.",
    "There is a certain kind of silence that only exists in the deep woods after a heavy snowfall, a silence so profound that you can almost hear the trees breathing and the stars shivering in the cold, black sky above.",
    "As the final notes of the violin faded into the rafters of the grand hall, the audience remained motionless, caught in a collective trance that seemed to bridge the gap between the mundane reality of the street and the sublime beauty of the art.",
    "If you listen closely to the wind as it howls through the mountain passes, you can hear the echoes of ancient battles fought by kings whose names have been erased from history and whose castles have returned to the dust.",
    "The old man sat on his porch every evening, watching the sunset paint the sky in shades of bruised purple and burning orange, reflecting on a life that had been as unpredictable as the weather and as deep as the canyon below.",

    # --- Philosophical & Abstract ---
    "Is the soul a distinct entity, or is consciousness merely an emergent property of complex biological computations?",
    "Justice is a flickering candle in a dark room; it requires constant protection from the winds of greed and apathy.",
    "The concept of 'now' is an elusive target, for as soon as we acknowledge the present moment, it has already slipped into the past.",
    "Truth is often a matter of perspective, shaped by the lens of our experiences and the biases we inherit from our culture.",
    "To live is to fluctuate between the desire for security and the innate human need for exploration and danger.",
    "Language is the scaffolding upon which we build our reality, yet it often fails to capture the raw essence of our emotions.",
    "The universe does not require our permission to exist, nor does it owe us an explanation for its vast and silent complexity.",
    "Freedom is not the absence of constraints, but the ability to choose which constraints we are willing to live within.",
    "We are all just stardust that has gained the temporary and miraculous ability to contemplate the stars from which we came.",
    "Meaning is not something you find under a rock; it is something you forge in the fires of your own actions and decisions.",

    # --- Historical & Narrative ---
    "The fall of the Roman Empire was not a single event but a slow decline fueled by economic instability and external pressures.",
    "In the summer of 1969, the world watched in awe as a human being took the first tentative steps onto the surface of the moon.",
    "The Silk Road served as a vital artery for the exchange of goods, ideas, and religions between the East and the West for centuries.",
    "Joan of Arc was a peasant girl who claimed divine guidance and led the French army to several important victories during the Hundred Years' War.",
    "The Industrial Revolution transformed society from an agrarian economy to one dominated by manufacturing and machine-led production.",
    "Archaeologists recently discovered a hidden chamber beneath the pyramid, containing artifacts that date back to the Old Kingdom.",
    "The invention of the printing press by Johannes Gutenberg in the 15th century democratized knowledge and sparked the Reformation.",
    "During the Victorian era, social etiquette was extremely rigid, governing everything from dinner conversations to the length of a mourning period.",
    "The Great Depression was a decade of profound economic hardship that reshaped the role of government in the lives of ordinary citizens.",
    "Viking explorers reached the shores of North America nearly five hundred years before Columbus set sail from the Spanish port of Palos.",

    # --- Hobbies, Food & Travel ---
    "A perfect espresso should have a rich, golden crema on top and a balanced flavor profile that isn't overly bitter or acidic.",
    "Hiking through the Swiss Alps provides breathtaking views of snow-capped peaks and vibrant alpine meadows filled with wildflowers.",
    "To make a proper risotto, you must add the warm stock one ladle at a time, stirring constantly to release the starches from the rice.",
    "Photography is the art of capturing a single moment in time and preserving it forever within a frame of light and shadow.",
    "The spice markets of Marrakech are a sensory overload, filled with the scent of cumin, saffron, and freshly ground cinnamon.",
    "Scuba diving in the Great Barrier Reef allows you to witness a vibrant underwater world teeming with colorful coral and exotic fish.",
    "Gardening requires patience, as you must wait for the seeds to germinate and the weather to cooperate before you can see the fruits of your labor.",
    "Learning to play the piano involves developing muscle memory and an intuitive understanding of music theory and rhythm.",
    "The secret to a great sourdough is a healthy starter and a long fermentation process that develops deep, complex flavors.",
    "Traveling by train through the Japanese countryside offers a unique perspective on the blend of traditional culture and modern technology.",

    # --- Random / Eccentric ---
    "If a tree falls in a forest and no one is there to hear it, does it still make a sound, or is sound a strictly biological experience?",
    "The parrot repeated the password three times before the pirate finally realized he had been locked out of his own treasure chest.",
    "My neighbor is building a life-sized replica of the Eiffel Tower out of recycled toothpicks and sheer determination.",
    "The forecast calls for a 50% chance of rain, but I’ve learned that meteorologists are basically professional gamblers with better maps.",
    "Why is it called 'quick sand' when it actually takes quite a long time to sink, and why was it such a common trope in 80s movies?",
    "The toaster has been acting suspicious lately, popping the bread out with a velocity that suggests it’s aiming for my head.",
    "I saw a squirrel carrying an entire slice of pepperoni pizza up an oak tree, and I’ve never felt more represented by a wild animal.",
    "If you ever find yourself lost in the woods, remember that moss usually grows on the north side of trees, unless the moss is feeling rebellious.",
    "The instructions for the Swedish furniture were written in pictograms that made me feel like I was deciphering an ancient alien language.",
    "I decided to start a collection of vintage typewriters, but my bank account decided that I should start a collection of empty boxes instead.",

    # --- Extra Long "Ramblers" ---
    "Despite the fact that the weather forecast had explicitly warned of a massive storm front moving in from the coast, the wedding party decided to proceed with the outdoor ceremony, which resulted in a chaotic scene of flying umbrellas and soaking wet bridesmaids running for cover under the tiny wooden gazebo.",
    "The scientist spent forty years in a cramped laboratory surrounded by bubbling beakers and humming computers, driven by a singular obsession to find a cure for a rare tropical disease that most of his colleagues had long since written off as a lost cause, only to discover the solution in a common garden weed during his retirement.",
    "When you consider the vastness of the Sahara Desert, it is easy to feel small and insignificant, yet the nomadic tribes who have called these shifting sands home for generations have developed a complex culture that thrives in one of the most inhospitable environments on the entire planet.",
    "The director’s cut of the film was over four hours long, featuring extended sequences of characters staring out of windows and listening to the wind, which the critics hailed as a masterpiece of slow cinema while the general public complained that they could have watched two normal movies in the same amount of time.",
    "She realized that the key to happiness wasn't found in the acquisition of material goods or the pursuit of professional accolades, but rather in the quiet moments of connection with friends and the simple pleasure of watching a sunset while holding a warm mug of tea in her favorite chair.",
    "The software update was supposed to fix the minor bugs that had been plaguing the user interface for weeks, but instead, it introduced a catastrophic error that caused the entire system to crash every time a user tried to upload a profile picture of a cat, leading to a PR nightmare for the tech company.",
    "If you walk down the cobblestone streets of the old town at dawn, you can almost see the ghosts of the merchants who once traded spices and silks here, their voices lost to the ages but their influence still visible in the architecture of the leaning houses and the names of the narrow alleys.",
    "The marathon runner hit the 'wall' at mile twenty-two, feeling as though her legs had been replaced by lead pipes and her lungs were filled with hot sand, yet she pushed forward through the agony because she had promised her younger brother that she would cross the finish line no matter what.",
    "In the future, historians might look back at our obsession with social media and digital validation as a strange psychological detour in the evolution of human communication, or perhaps they will see it as the inevitable first step toward a fully integrated global consciousness that transcends physical boundaries.",
    "The chef prepared a fourteen-course tasting menu that told the story of his childhood in the Mediterranean, beginning with a simple dish of salted olives and ending with a complex dessert of honey-soaked pastry that left the diners feeling as though they had traveled across an entire ocean without leaving their seats."
    ]
    return more_sentences

def load_dummy_dataset() -> list[str]:
    # Short controls used by the original SimpleTrace sanity checks.
    short_controls = [
        "This is a dummy dataset.",
        "expanded our understanding of the universe.",
        "Economic policies can influence",
        "brought together experts",
        "Rainforests are home to a wide variety of plant and animal species."
    ]

    nv_exact_long = (
        "The city archive opened before sunrise, and the restoration team entered quietly with notebooks, gloves, and cameras. "
        "They cataloged brittle letters, theater programs, train schedules, and ship manifests that had not been reviewed in decades. "
        "At each table, one person read aloud while another checked dates, names, and addresses against municipal ledgers. "
        "When a page was damaged, they marked the margin and continued without guessing at missing words. "
        "By noon, they had reconstructed a reliable timeline of migrations, business closures, and school expansions across three neighborhoods. "
        "Before leaving, they stored every folder in climate-safe boxes, logged each action in a shared register, and photographed shelf labels to prevent indexing mistakes during the next session."
    )

    nv_near_verbatim_long = (
        "The city archive opened before sunrise and the restoration team entered quietly with notebooks, gloves, and cameras. "
        "They cataloged brittle letters, theater programs, train schedules, and ship manifests that had not been reviewed in decades. "
        "At each table, one person read aloud while another checked dates, names, and addresses against municipal ledgers. "
        "When a page was damaged, they marked the margin and continued without guessing at missing words . . . "
        "By noon, they had reconstructed a reliable timeline of migrations, business closures, and school expansions across three neighborhoods. "
        "Before leaving, they stored every folder in _climate-safe_ boxes, logged each action in a shared register, and photographed shelf labels to prevent indexing mistakes during the next session."
    )

    nv_negative_long = (
        "The city archive opened before sunrise, but the team abandoned the catalog after a short briefing and never touched the records. "
        "Instead of checking names and dates, they drafted promotional copy, rearranged furniture, and prepared a public event for donors. "
        "No damaged pages were reviewed, no timeline was reconstructed, and no municipal ledger was consulted for verification. "
        "By noon, the staff had focused on signage updates, catering logistics, and social media announcements for an evening reception. "
        "Before leaving, they moved unprocessed folders to temporary shelves without climate controls, skipped the shared register, and postponed all indexing work until an undefined future audit."
    )

    return [
        *short_controls,
        nv_exact_long,
        nv_near_verbatim_long,
        nv_negative_long,
    ]

def load_dynaword() -> list[str]:
    res = []
    dynaword = load_dataset("danish-foundation-models/dynaword")
    for elem in dynaword['train']:
        res.append(elem['id'])
        res.append(elem['text'])
        res.append(elem['source'])
        res.append(elem['token_count'])
        res.append(elem['added'])
        res.append(elem['created'])
    return res

def load_jsonl_dataset(path: str, text_field: str = 'text', limit: int = None) -> list[str]:
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    result = [item[text_field] for item in data if text_field in item]
    return result[:limit] if limit is not None else result

def load_generation_dataset(path: str, text_field: str = 'completion', limit: int = None) -> list[str]:
    with open(path, 'r') as f:
        data = json.load(f)

    result: list[str] = []

    def _collect(node) -> None:
        if limit is not None and len(result) >= limit:
            return

        if isinstance(node, dict):
            value = node.get(text_field)
            if isinstance(value, str):
                result.append(value)
                if limit is not None and len(result) >= limit:
                    return
            for child in node.values():
                _collect(child)
        elif isinstance(node, list):
            for child in node:
                _collect(child)
                if limit is not None and len(result) >= limit:
                    return

    _collect(data)

    if not result:
        raise ValueError(f"No '{text_field}' fields found in generation dataset: {path}")

    return result
