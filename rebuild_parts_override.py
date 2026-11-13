"""
Rebuilds nisarg_shared/parts_override.json. The original lists were shuffled relative to the
prompts (see diagnose_shuffle.py), so the 250 lists below were rewritten from the master_suite.json prompts.

  python rebuild_parts_override.py            # dry run: validate and print overlap per category
  python rebuild_parts_override.py --apply    # back up the old file to .shuffled.bak and write
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

PO = Path("nisarg_shared/parts_override.json")
MS = Path("master_suite.json")

PARTS = {
    # --- animal_domestic 020-039 ---
    "020": ["head", "body", "pointed ears", "long tail"],
    "021": ["head", "body", "floppy ears", "tail"],
    "022": ["head", "body", "flowing mane", "tail"],
    "023": ["head", "body", "long ears", "tail"],
    "024": ["head", "body", "horns", "legs"],
    "025": ["head", "body", "curly tail", "legs"],
    "026": ["head", "comb", "body", "tail feathers"],
    "027": ["head", "body", "wings", "webbed feet"],
    "028": ["head", "domed shell", "legs", "tail"],
    "029": ["head", "body", "fins", "tail"],
    "030": ["head", "woolly body", "legs"],
    "031": ["head", "curved horns", "beard", "body"],
    "032": ["head", "arched body", "legs", "tail"],
    "033": ["head", "body", "front paws"],
    "034": ["head", "wattle", "body", "fanned tail"],
    "035": ["head", "long neck", "body", "legs"],
    "036": ["head", "large ears", "mane", "body"],
    "037": ["head", "body", "folded ears", "legs"],
    "038": ["head", "body", "legs", "tail"],
    "039": ["head", "body", "fanned tail feathers"],
    # --- animal_wild 040-059 ---
    "040": ["head", "body", "outstretched wings", "tail"],
    "041": ["head", "trunk", "tusks", "large ears"],
    "042": ["head", "mane", "body", "tail"],
    "043": ["head", "body", "fins", "tail"],
    "044": ["head", "body", "flippers", "tail fluke"],
    "045": ["head", "long neck", "body", "legs"],
    "046": ["head", "body", "legs", "tail"],
    "047": ["head", "flared hood", "coiled body"],
    "048": ["head", "mandibles", "body", "legs"],
    "049": ["head", "body", "hairy legs"],
    "050": ["head", "body", "legs", "paws"],
    "051": ["head", "body", "arms", "legs"],
    "052": ["head", "body", "legs", "tail"],
    "053": ["head", "body", "legs", "curled tail"],
    "054": ["head", "body", "triangular fins", "tail"],
    "055": ["head", "horn", "body", "legs"],
    "056": ["body", "pincers", "segmented tail", "legs"],
    "057": ["head", "body", "legs", "padded feet"],
    "058": ["head", "colored snout", "body", "limbs"],
    "059": ["head", "body", "wings", "talons"],
    # --- architecture 060-079 ---
    "060": ["main keep", "corner towers", "curtain walls", "drawbridge"],
    "061": ["columns", "pediment", "entablature", "stepped base"],
    "062": ["arch span", "deck", "abutments"],
    "063": ["tower", "light housing", "rocky base"],
    "064": ["tiered roofs", "body", "base"],
    "065": ["tiered arches", "outer wall", "arena floor"],
    "066": ["triangular panels", "base ring"],
    "067": ["log walls", "roof", "chimney", "porch"],
    "068": ["water tank", "support legs", "ladder"],
    "069": ["tower", "cables", "deck"],
    "070": ["arches", "piers", "water channel"],
    "071": ["terraced levels", "stairway", "top temple"],
    "072": ["iron frame", "glass panels", "roof vents"],
    "073": ["tower body", "sails", "cap"],
    "074": ["concrete facade", "windows", "roof"],
    "075": ["pillars", "crossbars"],
    "076": ["stone well", "roof", "winch"],
    "077": ["facade", "rose window", "towers", "portal"],
    "078": ["watchtower body", "crenellations", "base wall"],
    "079": ["dome", "entrance tunnel"],
    # --- electronics 080-099 ---
    "080": ["screen panel", "keyboard base", "hinge"],
    "081": ["screen", "body", "speaker grille"],
    "082": ["body", "lens", "grip"],
    "083": ["case", "front panel", "drive bays"],
    "084": ["headband", "ear cups"],
    "085": ["screen", "body", "camera bump"],
    "086": ["case", "keycaps", "base"],
    "087": ["body", "thumbsticks", "buttons", "d-pad"],
    "088": ["body", "spools", "control panel"],
    "089": ["base", "rotary dial", "handset"],
    "090": ["lamp shade", "articulating arm", "base"],
    "091": ["case", "lid", "earbuds"],
    "092": ["body", "antenna", "speaker grille"],
    "093": ["body", "lens", "pop-up flash"],
    "094": ["watch face", "body", "strap"],
    "095": ["microphone", "stand", "weighted base"],
    "096": ["scanner bed", "hinged lid"],
    "097": ["headset body", "cameras", "face gasket", "strap"],
    "098": ["body", "display", "buttons", "solar panel"],
    "099": ["hub body", "ports"],
    # --- fine_detail 100-119 ---
    "100": ["cranium", "jaw"],
    "101": ["helmet dome", "visor", "neck guard"],
    "102": ["frame", "vertical bars", "fleur-de-lis finials"],
    "103": ["chambers", "arteries", "veins"],
    "104": ["wood case", "clock face", "pendulum"],
    "105": ["board", "chips", "capacitors"],
    "106": ["core", "scales", "stem"],
    "107": ["head", "horns", "jaw", "teeth"],
    "108": ["frame", "filigree wirework", "clasp"],
    "109": ["palm", "fingers", "thumb"],
    "110": ["watch case", "watch face", "gear mechanism"],
    "111": ["helmet dome", "horsehair crest", "cheek guards"],
    "112": ["leaf blade", "veins", "stem"],
    "113": ["helix", "concha", "lobe"],
    "114": ["frame moulding", "foliate corner ornaments", "inner rabbet"],
    "115": ["main stalk", "branches", "polyp tubes"],
    "116": ["coiled shell", "chambers", "aperture"],
    "117": ["movement plate", "balance wheel", "gear train"],
    "118": ["perforated body", "scalloped border"],
    "119": ["vertebrae", "intervertebral discs"],
    # --- furniture 120-139 ---
    "120": ["seat", "legs", "curved backrest"],
    "121": ["seat cushions", "backrest", "metal legs"],
    "122": ["glass top", "tapered legs"],
    "123": ["shelves", "side panels", "back panel"],
    "124": ["headboard", "mattress platform", "frame"],
    "125": ["seat", "backrest", "curved runners", "armrests"],
    "126": ["tabletop", "adjustable legs", "crossbar"],
    "127": ["case", "drawers", "handles"],
    "128": ["steps", "side rails", "spreader"],
    "129": ["seat", "backrest", "cantilever frame"],
    "130": ["circular seat", "vertical legs", "footrest ring"],
    "131": ["fabric seat", "backrest", "x-frame legs"],
    "132": ["seat", "backrest", "padded armrests", "wooden legs"],
    "133": ["cabinet body", "hinged doors", "hanging rail"],
    "134": ["cabinet body", "glass doors", "shelves"],
    "135": ["seat sections", "backrests", "cushions"],
    "136": ["seat", "back slats", "armrests", "legs"],
    "137": ["tabletop", "drawer", "tapered legs"],
    "138": ["top surface", "lower shelves", "side panels"],
    "139": ["tilted tabletop", "legs", "parallel ruler"],
    # --- musical_instrument 160-179 ---
    "160": ["body", "neck", "headstock", "strings"],
    "161": ["body", "lid", "keyboard", "legs"],
    "162": ["body", "neck", "strings", "bow"],
    "163": ["body", "valves", "bell"],
    "164": ["drum shells", "cymbals", "hardware stands"],
    "165": ["soundboard", "curved neck", "pillar", "strings"],
    "166": ["coiled tubing", "valves", "bell"],
    "167": ["body", "neck", "strings", "endpin"],
    "168": ["pipes", "wind chest", "case"],
    "169": ["wooden bars", "resonator tubes", "frame"],
    "170": ["white keys", "black keys", "key bed"],
    "171": ["pear-shaped body", "neck", "pegbox", "strings"],
    "172": ["mouthpiece end", "conical body", "bell end"],
    "173": ["bag", "drones", "chanter", "blowpipe"],
    "174": ["playing surface", "skirt", "stand"],
    "175": ["gourd body", "neck", "frets", "strings"],
    "176": ["body", "keys", "curved neck", "bell"],
    "177": ["wooden bars", "frame", "resonators"],
    "178": ["mouthpiece", "slide", "bell"],
    "179": ["trapezoidal soundboard", "strings", "mallets"],
    # --- nature_plant 180-199 ---
    "180": ["trunk", "drooping branches", "needle canopy"],
    "181": ["petals", "stem", "thorns", "leaves"],
    "182": ["main column", "arms", "ribs"],
    "183": ["cap", "gills", "stalk"],
    "184": ["trunk", "branches", "canopy"],
    "185": ["seed head", "petals", "stem", "leaves"],
    "186": ["trap lobes", "stem", "base"],
    "187": ["grape cluster", "stem", "leaves"],
    "188": ["rind", "segments", "core"],
    "189": ["rachis", "leaflets", "stem"],
    "190": ["branch", "blossoms", "buds"],
    "191": ["fruit body", "leaf crown"],
    "192": ["outer petals", "inner petals", "center", "stem"],
    "193": ["trunk", "prop roots", "canopy"],
    "194": ["swollen trunk", "branches", "sparse canopy"],
    "195": ["pointed leaves", "central base"],
    "196": ["seed head", "pappus filaments", "stem"],
    "197": ["bananas", "curved stem"],
    "198": ["pitcher tube", "lid", "peristome rim", "stem"],
    "199": ["bark", "growth rings", "core"],
    # --- tool_hardware 220-239 ---
    "220": ["metal claw head", "wooden handle"],
    "221": ["fixed jaw", "adjustable jaw", "handle"],
    "222": ["body", "chuck", "trigger", "battery pack"],
    "223": ["serrated blade", "wooden handle"],
    "224": ["jaws", "pivot", "handles"],
    "225": ["bolt head", "threaded shaft", "hex nut"],
    "226": ["fixed jaw", "sliding jaw", "screw handle", "mounting clamp"],
    "227": ["frame", "bubble vials"],
    "228": ["housing", "metal tape", "hook"],
    "229": ["body", "blade", "tote handle"],
    "230": ["ratchet head", "handle", "socket"],
    "231": ["handle", "shaft", "bit"],
    "232": ["fixed jaw", "hook jaw", "handle"],
    "233": ["body", "blade", "slider"],
    "234": ["folding holder", "hex keys"],
    "235": ["cutting jaws", "gauge notches", "handles"],
    "236": ["main scale", "sliding jaw", "depth probe"],
    "237": ["box body", "guide slots"],
    "238": ["hexagonal shank", "cutting edge"],
    "239": ["body", "handle", "loading channel"],
    # --- vehicle_air_water 240-259 ---
    "240": ["fuselage", "wings", "engines", "tail"],
    "241": ["fuselage", "wings", "propeller", "tail"],
    "242": ["fuselage", "swept wings", "tail fins", "canopy"],
    "243": ["cabin", "main rotor", "tail boom", "tail rotor"],
    "244": ["envelope", "basket", "burner"],
    "245": ["orbiter", "external tank", "solid rocket boosters"],
    "246": ["hull", "masts", "sails"],
    "247": ["hull", "windshield", "outboard motor"],
    "248": ["hull", "conning tower", "propeller"],
    "249": ["hull", "dragon prow", "oars", "sail"],
    "250": ["fuselage", "stacked wings", "propeller", "landing gear"],
    "251": ["fuselage", "wings", "pontoon floats", "propeller"],
    "252": ["envelope", "gondola", "fins"],
    "253": ["hulls", "deck", "mast"],
    "254": ["hull", "foil struts", "cabin"],
    "255": ["hull", "superstructure", "deck cranes"],
    "256": ["fuselage", "rotor disc", "wings", "propeller"],
    "257": ["hull", "cockpit", "paddle"],
    "258": ["inflatable hull", "console", "outboard motors"],
    "259": ["canopy", "suspension lines", "harness"],
    # --- vehicle_land 260-279 ---
    "260": ["body", "wheels", "headlights"],
    "261": ["frame", "wheels", "handlebars", "suspension fork"],
    "262": ["body", "wheels", "fairing", "exhaust pipes"],
    "263": ["lower deck", "upper deck", "wheels"],
    "264": ["cab", "cargo bed", "wheels"],
    "265": ["boiler", "driving wheels", "smokestack", "cab"],
    "266": ["chassis", "wheels", "engine", "roll bar"],
    "267": ["body", "wheels", "rear wing", "front wing"],
    "268": ["body", "wheels", "headlights"],
    "269": ["hull", "turret", "tracks", "gun barrel"],
    "270": ["large front wheel", "small rear wheel", "frame", "seat"],
    "271": ["body", "rear wheels", "front wheels", "loader bucket"],
    "272": ["cab", "body", "aerial ladder", "outriggers"],
    "273": ["cabin", "canopy", "wheels"],
    "274": ["carriage body", "spoked wheels", "coachman bench"],
    "275": ["cab", "body", "tracks"],
    "276": ["cab", "mixing drum", "chute", "wheels"],
    "277": ["cab", "dump bed", "pivot joint", "tires"],
    "278": ["car body", "pantograph", "bogies"],
    "279": ["platform", "wheels", "handlebar"],
    # --- original_visuals 280-309 ---
    "280": ["fuselage", "dragonfly wings", "tail"],
    "281": ["head", "buttoned vest", "trousers", "ears"],
    "282": ["chest body", "lid", "gold lock", "corner brackets"],
    "283": ["seat", "backrest", "tapered legs"],
    "284": ["wagon bed", "spoked wheels", "canvas cover"],
    "285": ["head", "glowing eyes", "body", "limbs"],
    "286": ["headband", "ear cups", "brass gears"],
    "287": ["head", "goggles", "shell", "legs"],
    "288": ["body", "off-road tires", "roll cage"],
    "289": ["head", "winter jacket", "backpack", "tail"],
    "290": ["log walls", "roof", "chimney", "crooked door"],
    "291": ["chest body", "lid", "crown", "star clasps"],
    "292": ["hull", "conning tower", "robot arms", "propeller"],
    "293": ["head", "body", "small horns", "legs"],
    "294": ["body", "wheels", "scorpion tail"],
    "295": ["helmet", "torso suit", "arms", "legs"],
    "296": ["armored hull", "wheels", "optical sensors"],
    "297": ["padlock body", "shackle", "neon strips"],
    "298": ["mug body", "handle", "iron bands"],
    "299": ["envelope", "gondola", "propellers"],
    "300": ["carapace", "tank-tread legs", "claws", "tail"],
    "301": ["horned helmet", "torso suit", "boots", "arms"],
    "302": ["chassis", "hover engines", "canopy"],
    "303": ["central keep", "corner towers", "walls", "gate"],
    "304": ["chariot body", "dragon wings", "wheels"],
    "305": ["chassis", "circular speakers", "brass dials", "handle"],
    "306": ["false front facade", "building body", "swinging doors", "porch"],
    "307": ["head", "torso", "slender limbs", "tail"],
    "308": ["wagon body", "tank treads", "canopy"],
    "309": ["body", "dorsal tank", "nozzles", "trigger"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    old = json.loads(PO.read_text())

    if set(PARTS) != set(old):
        missing = sorted(set(old) - set(PARTS))
        extra = sorted(set(PARTS) - set(old))
        sys.exit(f"KEY MISMATCH  missing={missing}  extra={extra}")

    bad = []
    for k, v in sorted(PARTS.items()):
        if not isinstance(v, list) or not (2 <= len(v) <= 4):
            bad.append((k, f"{len(v) if isinstance(v, list) else v} parts"))
        elif any((not isinstance(x, str)) or not x.strip() for x in v):
            bad.append((k, "empty/non-str part"))
        elif len({x.lower() for x in v}) != len(v):
            bad.append((k, "duplicate part"))
        elif any(w in {"left", "right"} for x in v for w in x.lower().split()):
            bad.append((k, "left/right token"))
    if bad:
        for k, why in bad:
            print(f"  BAD {k}: {why}")
        sys.exit(f"{len(bad)} invalid entries")

    if MS.exists():
        ms = json.loads(MS.read_text())
        prm, cats = ms["prompts"], ms["categories"]
        tok = lambda s: {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) > 2}
        agg = defaultdict(lambda: [0, 0.0])
        for k, v in PARTS.items():
            i = int(k)
            agg[cats[i]][0] += 1
            agg[cats[i]][1] += len(tok(" ".join(v)) & tok(prm[i]))
        print(f"{'category':22}{'n':>4}{'mean prompt-overlap':>22}   (was ~0.4 when shuffled)")
        for c, (n, s) in sorted(agg.items()):
            print(f"{c:22}{n:>4}{s / n:>22.2f}")

    print(f"\n{len(PARTS)} entries, key set matches, all shape checks pass.")
    if not args.apply:
        print("dry run -- rerun with --apply to write")
        return

    bak = PO.with_name(PO.name + ".shuffled.bak")
    if not bak.exists():
        bak.write_text(PO.read_text())
        print(f"backed up original -> {bak}")
    PO.write_text(json.dumps({k: PARTS[k] for k in sorted(PARTS, key=int)}, indent=2) + "\n")
    print(f"wrote {PO}")


if __name__ == "__main__":
    main()
