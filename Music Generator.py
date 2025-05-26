import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import pygame
from pygame import mixer
import threading
import tkinter.font as tkfont
from music21 import chord, stream, note, tempo, instrument, midi, harmony, meter, articulations
import random
import os

class GenreSelection:
    def __init__(self, root):
        self.root = root
        self.selected_genre = tk.StringVar()
        
        # Black with red accents
        self.bg_color = "#000000"         # Black background

        self.font_large = ('Figtree', 24, 'bold') if 'Figtree' in tkfont.families() else ('Helvetica', 24, 'bold')
        self.font_medium = ('Figtree', 14, 'bold') if 'Figtree' in tkfont.families() else ('Helvetica', 14, 'bold')
        self.font_small = ('Figtree', 12, 'bold') if 'Figtree' in tkfont.families() else ('Helvetica', 10)

        self.card_color = "#1a1a1a"       # Darker gray for cards
        self.text_color = "#ffffff"       # White text
        self.accent_color = "#c51017"     # Red accent
        self.highlight_color = "#ff4d4d"  # Light red for highlights

        self.setup_ui()
    
    def setup_ui(self):
        self.root.title("Music Genre Selector")
        self.root.geometry("800x800")
        self.root.configure(bg=self.bg_color)
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=30)
        header_frame.pack(fill=tk.X)
        
        tk.Label(
            header_frame,
            text="Select Your Genre",
            font=self.font_large,
            fg=self.accent_color,
            bg=self.bg_color
        ).pack()
        
        # Genre selection cards
        genres_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        genres_frame.pack(fill=tk.BOTH, expand=True)
        
        # Genre images (placeholder colors)
        genre_data = [
            ("Jazz", "🎷"),
            ("HipHop", "🎤"),
            ("Classical", "🎻"),
            ("Rock", "🎸")
        ]
    
        self.genre_cards = []
        for genre, emoji in genre_data:
            card = tk.Frame(
                genres_frame,
                bg=self.card_color,
                padx=15,
                pady=15,
                relief=tk.FLAT,
                borderwidth=2,
                highlightbackground=self.card_color,
                highlightcolor=self.highlight_color
            )
            card.pack(fill=tk.X, pady=10)

            rb = tk.Radiobutton(
                card,
                text=f"{emoji}  {genre}",
                variable=self.selected_genre,
                value=genre.lower(),
                font=self.font_medium,
                bg=self.card_color,
                fg=self.text_color,
                selectcolor=self.card_color,
                activebackground=self.card_color,
                activeforeground=self.highlight_color,
                highlightthickness=2,
                indicatoron=0,
                width=30,
                anchor="center",
                padx=20,
                pady=10,
                bd=0,
                relief=tk.FLAT
            )
            rb.pack(fill=tk.X)

            rb.config(command=lambda c=card: self.highlight_card(c))
            self.genre_cards.append((card, rb))

        button_frame = tk.Frame(self.root, bg=self.bg_color, pady=20)
        button_frame.pack(fill=tk.X)
        
        tk.Button(
            button_frame,
            text="Generate Music",
            font=self.font_small,
            bg=self.accent_color,
            fg="white",
            activebackground=self.highlight_color,
            command=self.launch_music_generator,
            padx=30,
            pady=10
        ).pack()
        
        self.center_window()
    
    def highlight_card(self, selected_card):
        for card, rb in self.genre_cards:
            if card == selected_card:
                card.config(highlightbackground=self.highlight_color, highlightthickness=2)
                rb.config(fg=self.highlight_color, font=(self.font_medium[0], self.font_medium[1], "bold"))
            else:
                card.config(highlightbackground=self.card_color, highlightthickness=1)
                rb.config(fg=self.text_color, font=self.font_medium)

    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def launch_music_generator(self):
        if not self.selected_genre.get():
            messagebox.showwarning("No Genre Selected", "Please select a genre first!")
            return
        
        self.root.destroy()
        main(self.selected_genre.get())

all_scales = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']

genre_scales = {
    'classical': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'rock': ['E', 'G', 'A', 'B', 'D'],
    'jazz': ['C', 'Eb', 'F', 'Gb', 'G', 'Bb'],
    'hiphop': ['C', 'D', 'Eb', 'G', 'A']
}

instrument_timbre = {
    'Piano': 4,
    'ElectricGuitar': 8,
    'Saxophone': 7,
    'Woodblock': 5,
    'Flute': 3,
    'Trumpet': 6
}

genre_chord_progressions = {
    'jazz': [
        ["Dm7", "G7", "Cmaj7"],
        ["Am7", "D7", "Gmaj7"]
    ],
    'classical': [
        ["C", "F", "G", "C"],
        ["D", "G", "A", "D"]
    ],
    'rock': [
        ["C", "G", "Am", "F"],
        ["E", "A", "B", "E"]
    ],
    'hiphop': [
        ["Cm", "A-", "E-", "B-"],
        ["Am", "F", "G", "Em"]
    ]
}

genre_rhythm_patterns = {
    'jazz': [0.5, 0.5, 1.0, 1.0, 0.5, 0.5],
    'classical': [1.0, 1.0, 0.5, 0.5, 2.0],
    'rock': [1.0, 1.0, 1.0, 1.0],
    'hiphop': [1.0, 0.5, 0.5, 1.0, 1.0]
}

def generate_random_beat(genre=None, duration_seconds=10, features=None):

    s = stream.Stream()

    # Random genre selection if not provided
    if genre is None:
        genre = random.choice(['classical', 'rock', 'jazz', 'hiphop'])

    s.append(meter.TimeSignature('4/4'))

    genre_randomness = {
        'classical': {
            'tempo_range': (60, 120),
            'velocity_range': (70, 110),
            'octave_variation': 0.3,
            'rhythm_variation': 0.2,
            'articulation': 0.1
        },
        'rock': {
            'tempo_range': (90, 150),
            'velocity_range': (90, 127),
            'octave_variation': 0.5,
            'rhythm_variation': 0.4,
            'articulation': 0.3
        },
        'jazz': {
            'tempo_range': (90, 140),
            'velocity_range': (80, 120),
            'octave_variation': 0.7,
            'rhythm_variation': 0.6,
            'articulation': 0.5
        },
        'hiphop': {
            'tempo_range': (80, 110),
            'velocity_range': (85, 115),
            'octave_variation': 0.4,
            'rhythm_variation': 0.3,
            'articulation': 0.2
        }
    }

    params = genre_randomness.get(genre)

    bpm = round(features['tempo']) if features and 'tempo' in features else random.randint(*params['tempo_range'])
    bpm += random.randint(-5, 5)
    s.append(tempo.MetronomeMark(number=max(40, min(200, bpm))))

    chords = random.choice(genre_chord_progressions[genre])
    base_rhythm = genre_rhythm_patterns[genre]

    rhythm = [dur * (1 + (random.random() - 0.5) * params['rhythm_variation']) for dur in base_rhythm]
    rhythm = [max(0.25, min(2.0, dur)) for dur in rhythm]

    if features and 'timbre' in features:
        closest_instr = min(instrument_timbre.items(), key=lambda x: abs(x[1] - features['timbre']))[0]
        instr_mapping = {
            'Piano': instrument.Piano(),
            'ElectricGuitar': instrument.ElectricGuitar(),
            'Saxophone': random.choice([
                instrument.SopranoSaxophone(),
                instrument.AltoSaxophone(),
                instrument.TenorSaxophone()
            ]),
            'Woodblock': instrument.Woodblock(),
            'Flute': instrument.Flute(),
            'Trumpet': instrument.Trumpet()
        }
        instr = instr_mapping.get(closest_instr, instrument.Piano())
    else:
        instr_options = {
            'jazz': [
                instrument.AltoSaxophone(),
                instrument.Trumpet(),
                instrument.Piano(),
                instrument.ElectricBass()
            ],
            'classical': [
                instrument.Piano(),
                instrument.Violin(),
                instrument.Flute(),
                instrument.Violoncello()
            ],
            'rock': [
                instrument.ElectricGuitar(),
                instrument.ElectricBass(),
                instrument.Piano()
            ],
            'hiphop': [
                instrument.ElectricBass(),
                instrument.Harpsichord(),
                instrument.Woodblock(),
                instrument.Percussion()
            ]
        }

        instr = random.choice(instr_options.get(genre, [instrument.Piano()]))

    s.insert(0, instr)

    total_beats = (duration_seconds * bpm) / 60
    current_time = 0
    chord_index = 0
    rhythm_index = 0

    while current_time < total_beats:
        dur = rhythm[rhythm_index % len(rhythm)]
        if current_time + dur > total_beats:
            break

        current_chord = chords[chord_index % len(chords)]
        ch_symbol = harmony.ChordSymbol(current_chord)

        chord_notes = []
        for p in ch_symbol.pitches:
            base_octave = 4 if p.name not in ['E', 'F', 'G'] else 5
            octave_variation = int((random.random() - 0.5) * 2 * params['octave_variation'])
            octave = max(3, min(6, base_octave + octave_variation))
            chord_notes.append(note.Note(p.name + str(octave)))

        ch = chord.Chord(chord_notes)
        ch.duration.quarterLength = dur

        base_velocity = random.randint(*params['velocity_range'])
        velocity_variation = random.randint(-10, 10)
        ch.volume.velocity = max(40, min(127, base_velocity + velocity_variation))

        if random.random() < params['articulation']:
            ch.articulations.append(
                articulations.Staccato() if random.random() > 0.5 else articulations.Tenuto()
            )

        s.insert(current_time, ch)

        if genre == 'rock':
            drum_part = stream.Part()
            drum_part.insert(0, instrument.Woodblock())

            beat_note = note.Note("C2")
            snare_note = note.Note("D2")
            hihat_note = note.Note("F#2")

            for i in range(int(dur * 4)):
                t = current_time + i * (dur / 4.0)
                if i % 4 == 0:
                    drum_part.insert(t, note.Note(beat_note.nameWithOctave))
                elif i % 4 == 2:
                    drum_part.insert(t, note.Note(snare_note.nameWithOctave))
                else:
                    hh = note.Note(hihat_note.nameWithOctave)
                    hh.volume.velocity = 60
                    drum_part.insert(t, hh)

            s.append(drum_part)

        current_time += dur
        rhythm_index += 1
        chord_index += 1

    return s, genre


def extract_features(beat_stream):
    notes = [n for n in beat_stream.notesAndRests]
    pitch_vals = [n.pitch.midi for n in notes if isinstance(n, note.Note)]
    sharp_count = sum(1 for n in notes if isinstance(n, note.Note) and '#' in n.name)

    avg_pitch = sum(pitch_vals) / len(pitch_vals) if pitch_vals else 0
    pitch_range = max(pitch_vals) - min(pitch_vals) if pitch_vals else 0
    density = len(notes)

    bpm = 100
    for el in beat_stream:
        if isinstance(el, tempo.MetronomeMark):
            bpm = el.number

    instr_name = next((i.classes[0] for i in beat_stream.getInstruments(returnDefault=True)), 'Piano')
    timbre_score = instrument_timbre.get(instr_name, 5)

    sharpness_ratio = sharp_count / len(pitch_vals) if pitch_vals else 0
    beat_density = density / 7

    return {
        'timbre': timbre_score,
        'texture': density,
        'avg_pitch': avg_pitch,
        'pitch_range': pitch_range,
        'tempo': bpm,
        'beat_density': beat_density,
        'sharpness': sharpness_ratio
    }

def fitness_function(genre, features):
    score = 0
    weights = []

    if genre == 'classical':
        weights = [
            (15, 3 <= features['timbre'] <= 5),
            (5, 3 <= features['timbre'] <= 4),
            (15, 64 <= features['avg_pitch'] <= 74),
            (5, 69 <= features['avg_pitch'] <= 74),
            (15, 18 <= features['pitch_range'] <= 30),
            (5, 25 <= features['pitch_range'] <= 30),
            (15, 50 <= features['texture'] <= 80),
            (5, 65 <= features['texture'] <= 75),
            (7, 90 <= features['tempo'] <= 120),
            (3, 100 <= features['tempo'] <= 110),
            (7, features['sharpness'] < 0.2),
            (3, features['sharpness'] < 0.1)
        ]

    elif genre == 'rock':
        weights = [
            (15, 7 <= features['timbre'] <= 9),
            (5, 8 <= features['timbre'] <= 9),
            (15, 52 <= features['avg_pitch'] <= 62),
            (5, 55 <= features['avg_pitch'] <= 59),
            (15, 15 <= features['pitch_range'] <= 25),
            (5, 17 <= features['pitch_range'] <= 22),
            (20, 55 <= features['texture'] <= 90),
            (20, 80 <= features['texture'] <= 90),
            (8, 110 <= features['tempo'] <= 150),
            (2, 130 <= features['tempo'] <= 150),
            (8, features['sharpness'] > 0.25),
            (2, features['sharpness'] > 0.3)
        ]

    elif genre == 'jazz':
        weights = [
            (20, 6 <= features['timbre'] <= 8),
            (20, 58 <= features['avg_pitch'] <= 78),
            (20, 18 <= features['pitch_range'] <= 35),
            (5, 22 <= features['pitch_range'] <= 30),
            (20, 60 <= features['texture'] <= 100),
            (10, 100 <= features['tempo'] <= 140),
            (10, 0.15 <= features['sharpness'] <= 0.35)
        ]

    elif genre == 'hiphop':
        weights = [
            (25, 4 <= features['timbre'] <= 6),
            (20, 50 <= features['avg_pitch'] <= 65),
            (20, 6 <= features['pitch_range'] <= 18),
            (20, 45 <= features['texture'] <= 75),
            (10, 80 <= features['tempo'] <= 100),
            (10, features['sharpness'] < 0.25)
        ]

    score = sum(weight for weight, condition in weights if condition)

    max_score = sum(weight for weight, _ in weights)

    # Normalize
    normalized_score = (score / max_score) * 100 if max_score > 0 else 0
    return normalized_score


def save_beat(s, filename):
    if not os.path.exists('output'):
        os.makedirs('output')
    filepath = os.path.join('output', filename)
    
    mf = midi.translate.streamToMidiFile(s)
    mf.open(filepath, 'wb')
    mf.write()
    mf.close()
    return filepath

def generate_initial_population(user_genre=None, num_beats=500):
    all_beats = []
    features_for_all = []
    genres = list(genre_scales.keys())

    for genre in genres:
        for j in range(10):
            s, _ = generate_random_beat(genre=genre)
            features = extract_features(s)
            score = fitness_function(genre, features)
            filename = f"{genre}_initial_beat_{j}.mid"
            save_beat(s, filename)
            all_beats.append((score, filename, features))
            features_for_all.append(features)

    remaining = num_beats - (10 * len(genres))
    for i in range(remaining):
        genre = user_genre if user_genre else random.choice(genres)
        s, _ = generate_random_beat(genre=genre)
        features = extract_features(s)
        score = fitness_function(genre, features)
        filename = f"random_beat_{i}.mid"
        save_beat(s, filename)
        all_beats.append((score, filename, features))
        features_for_all.append(features)

    return all_beats, features_for_all

def select_best(all_beats, user_genre):
    parents = sorted(all_beats, key=lambda x: x[0], reverse=True)[:2]
    
    print(f"\n Top 2 matching beats for genre '{user_genre}':")
    for score, filename, _ in parents:
        print(f"  {filename} (Fitness score: {score})")

    return parents

def get_mid_value_by_range(val1, val2, feature_range):
    avg_range = (feature_range[0] + feature_range[1]) / 2
    return val1 if abs(val1 - avg_range) < abs(val2 - avg_range) else val2

itr_for_post_selection = 0

child_beats = []
child_features = []

def crossover(parents, genre, all_beats, features_for_all):
    if len(parents) < 2:
        return all_beats, features_for_all

    beat1 = parents[0]
    beat2 = parents[1]
    features1 = beat1[2]
    features2 = beat2[2]

    genre_ranges = {
        'classical': {
            'timbre': (3, 5), 'avg_pitch': (64, 74), 'pitch_range': (18, 30),
            'texture': (50, 80), 'tempo': (90, 120), 'sharpness': (0, 0.2)
        },
        'hiphop': {
            'timbre': (4, 6), 'avg_pitch': (50, 65), 'pitch_range': (6, 18),
            'texture': (45, 75), 'tempo': (80, 100), 'sharpness': (0, 0.25)
        },
        'jazz': {
            'timbre': (6, 8), 'avg_pitch': (58, 78), 'pitch_range': (18, 35),
            'texture': (60, 100), 'tempo': (100, 140), 'sharpness': (0.15, 0.35)
        },
        'rock': {
            'timbre': (7, 9), 'avg_pitch': (52, 62), 'pitch_range': (15, 90),
            'texture': (55, 90), 'tempo': (110, 150), 'sharpness': (0.25, 1)
        }
    }

    selection_strategy = {
        'classical': {
            'timbre': 'min', 'avg_pitch': 'mid', 'pitch_range': 'max',
            'texture': 'mid', 'tempo': 'mid', 'sharpness': 'min'
        },
        'hiphop': {
            'timbre': 'mid', 'avg_pitch': 'min', 'pitch_range': 'min',
            'texture': 'mid', 'tempo': 'min', 'sharpness': 'min'
        },
        'jazz': {
            'timbre': 'max', 'avg_pitch': 'max', 'pitch_range': 'max',
            'texture': 'max', 'tempo': 'mid', 'sharpness': 'mid'
        },
        'rock': {
            'timbre': 'max', 'avg_pitch': 'mid', 'pitch_range': 'mid',
            'texture': 'max', 'tempo': 'max', 'sharpness': 'max'
        }
    }

    best_features = {}
    for feature in selection_strategy[genre]:
        strategy = selection_strategy[genre][feature]
        val1 = features1[feature]
        val2 = features2[feature]

        if strategy == 'min':
            best_features[feature] = min(val1, val2)
        elif strategy == 'max':
            best_features[feature] = max(val1, val2)
        elif strategy == 'mid':
            best_features[feature] = get_mid_value_by_range(val1, val2, genre_ranges[genre][feature])

    for i in range(3):
        child_stream, _ = generate_random_beat(features=best_features)
        child_filename = f"child_beat_{i}_from_top2.mid"
        child_score = fitness_function(genre, best_features)
        save_beat(child_stream, child_filename)

        child_beats.append((child_score, child_filename, best_features))
        child_features.append(best_features)

    # Add remaining beats
    remaining_beats = [b for b in all_beats if b not in parents[:2]]
    child_beats.extend(remaining_beats)
    child_features.extend([b[2] for b in remaining_beats])

    return child_beats, child_features

def mutate_beats(child_beats, all_features, genre, num_mutations=3):
    mutated_beats = []
    
    if not child_beats:
        raise ValueError("No child beats provided for mutation")
    if not all_features:
        raise ValueError("No features available for reference")

    feature_keys = ['timbre', 'texture', 'avg_pitch', 'pitch_range', 'tempo', 'sharpness']
    
    for _ in range(num_mutations):

        original_beat = random.choice(child_beats)
        mutated_features = dict(original_beat[2])
        
        reference = random.choice(all_features)

        num_features_to_mutate = random.randint(1, len(feature_keys))
        features_to_mutate = random.sample(feature_keys, num_features_to_mutate)
        
        for feature in features_to_mutate:
            mutation_type = random.choice(['blend', 'randomize', 'nudge'])
            
            if mutation_type == 'blend':
                blend_factor = random.random()
                mutated_value = (blend_factor * reference[feature] + 
                               (1 - blend_factor) * mutated_features[feature])
            elif mutation_type == 'randomize':
                if feature == 'timbre':
                    mutated_value = random.uniform(3, 9)
                elif feature == 'texture':
                    mutated_value = random.randint(30, 120)
                elif feature == 'avg_pitch':
                    mutated_value = random.randint(40, 80)
                elif feature == 'pitch_range':
                    mutated_value = random.randint(5, 40)
                elif feature == 'tempo':
                    mutated_value = random.randint(60, 180)
                elif feature == 'sharpness':
                    mutated_value = random.uniform(0, 0.5)
            else:  # nudge
                # Small adjustment to current value
                nudge_amount = random.uniform(-0.2, 0.2)
                if feature in ['timbre', 'sharpness']:
                    mutated_value = mutated_features[feature] + nudge_amount
                else:
                    mutated_value = mutated_features[feature] * (1 + nudge_amount)

            if feature == 'timbre':
                mutated_value = max(0, min(10, mutated_value))
            elif feature == 'texture':
                mutated_value = max(10, min(200, round(mutated_value)))
            elif feature == 'avg_pitch':
                mutated_value = max(20, min(100, round(mutated_value)))
            elif feature == 'pitch_range':
                mutated_value = max(5, min(50, round(mutated_value)))
            elif feature == 'tempo':
                mutated_value = max(40, min(200, round(mutated_value)))
            elif feature == 'sharpness':
                mutated_value = max(0, min(1, mutated_value))
            
            mutated_features[feature] = mutated_value

        # Generate and save mutated beat
        mutated_stream, _ = generate_random_beat(features=mutated_features)
        filename = f"mutated_beat_{random.randint(1000, 9999)}.mid"
        score = fitness_function(genre, mutated_features)
        save_beat(mutated_stream, filename)

        mutated_beats.append((score, filename, mutated_features))

    return mutated_beats

def post_selection(new_beats):
    global all_beats, itr_for_post_selection

    n = 50  # truncation after n = 50

    all_beats = sorted(all_beats, key=lambda x: x[0])  # ascending sort

    if itr_for_post_selection < n:
        for i in range(min(4, len(new_beats))):
            all_beats[i] = new_beats[i]
    else:
        num_to_keep = int(len(all_beats) * 0.6)
        new_beats = sorted(all_beats, key=lambda x: x[0], reverse=True)[:num_to_keep]

        for beat in new_beats:
            all_beats.append(beat)

    all_beats = sorted(all_beats, key=lambda x: x[0], reverse=True)
    itr_for_post_selection += 1
    return all_beats

def update_scores(population, genre):
    updated_population = []
    for beat in population:
        new_score = fitness_function(genre, beat[2])
        updated_population.append((new_score, beat[1], beat[2]))
    return updated_population

def select_final_best(all_beats, user_genre):
    best_beat = sorted(all_beats, key=lambda x: x[0], reverse=True)[0]
    
    print(f"\n Best beat generated for '{user_genre}' is:")
    print(f"  {best_beat[1]} (Fitness score: {best_beat[0]})")
    
    return [best_beat]

class MusicPlayerGUI:
    def __init__(self, root, beats, genre):
        self.root = root
        self.beats = beats
        self.genre = genre
        self.current_playing = None
        mixer.init()
        self.equalizer_running = False

        # Black with red accents
        self.bg_color = "#000000"

        self.font_large = ('Figtree', 24, 'bold') if 'Figtree' in tkfont.families() else ('Helvetica', 24, 'bold')
        self.font_medium = ('Figtree', 14, 'bold') if 'Figtree' in tkfont.families() else ('Helvetica', 14, 'bold')
        self.font_small = ('Figtree', 12, 'bold') if 'Figtree' in tkfont.families() else ('Helvetica', 10)

        self.card_color = "#1a1a1a"
        self.text_color = "#ffffff"
        self.accent_color = "#990000"
        self.button_color = "#990000"

        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("Evolutionary Music Generator")
        self.root.geometry("800x600")
        self.root.configure(bg=self.bg_color)
        
        self.root.minsize(600, 400)
        
        header_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame,
            text=f"Top {self.genre.capitalize()} Beat",
            font=self.font_large,
            fg=self.accent_color,
            bg=self.bg_color
        )
        title_label.pack(side=tk.LEFT)
        
        try:
            eq_img = Image.new('RGB', (100, 30), color=self.bg_color)
            self.eq_photo = ImageTk.PhotoImage(eq_img)
            self.eq_label = tk.Label(header_frame, image=self.eq_photo, bg=self.bg_color)
            self.eq_label.pack(side=tk.RIGHT)
        except:
            pass
        
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.beat_widgets = []
        for i, (score, filename, _) in enumerate(self.beats[:2]):
            card = tk.Frame(
                main_frame,
                bg=self.card_color,
                padx=20,
                pady=15,
                relief=tk.RAISED,
                borderwidth=2,
                highlightbackground=self.accent_color
            )
            card.pack(fill=tk.X, pady=15, ipady=10)
            
            info_frame = tk.Frame(card, bg=self.card_color)
            info_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(
                info_frame,
                text=f"• Score: {score:.1f}",
                font=self.font_small,
                fg=self.accent_color,
                bg=self.card_color
            ).pack(side=tk.LEFT)
            
            tk.Label(
                info_frame,
                text=filename,
                font=self.font_small,
                fg=self.text_color,
                bg=self.card_color
            ).pack(side=tk.LEFT, padx=20)

            controls_frame = tk.Frame(card, bg=self.card_color)
            controls_frame.pack(fill=tk.X)
            
            play_btn = tk.Button(
                controls_frame,
                text="▶ Play",
                font=self.font_small,
                bg=self.button_color,
                fg="white",
                activebackground=self.accent_color,
                borderwidth=0,
                padx=15,
                pady=5,
                command=lambda f=filename: self.play_beat(f)
            )
            play_btn.pack(side=tk.LEFT, padx=5)
            
            stop_btn = tk.Button(
                controls_frame,
                text="■ Stop",
                font=self.font_small,
                bg="#262626",
                fg="white",
                activebackground="#ff4d4d",
                borderwidth=0,
                padx=15,
                pady=5,
                command=self.stop_beat
            )
            stop_btn.pack(side=tk.LEFT)
            
            self.beat_widgets.append((card, play_btn, stop_btn))
        
        footer_frame = tk.Frame(self.root, bg=self.bg_color, pady=20)
        footer_frame.pack(fill=tk.X)
        
        tk.Button(
            footer_frame,
            text="Exit",
            font=self.font_small,
            bg="#990000",
            fg="white",
            activebackground="#D08770",
            command=self.root.quit,
            padx=20,
            pady=5
        ).pack(side=tk.RIGHT, padx=20)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#3B4252",
            fg=self.text_color,
            anchor=tk.W,
            font=self.font_small
        ).pack(fill=tk.X, side=tk.BOTTOM)
    
    def animate_equalizer(self, label):
        """Simple animation for the equalizer graphic while music is playing"""
        if not self.equalizer_running:
            return

        colors = [self.accent_color, "#cc0000", "#f14a51", "#ed1824", "#f47a7f", "#960c12"]
        try:
            img = Image.new('RGB', (100, 30), color=self.bg_color)
            draw = ImageDraw.Draw(img)

            # Draw random bars for equalizer effect
            for i in range(5):
                h = random.randint(5, 25)
                draw.rectangle([i*20, 30-h, i*20+15, 30], fill=colors[i % 3])

            self.eq_photo = ImageTk.PhotoImage(img)
            label.config(image=self.eq_photo)
            self.root.after(200, lambda: self.animate_equalizer(label))
        except:
            pass

    def play_beat(self, filename):
        def play():
            if self.current_playing:
                mixer.music.stop()
            try:
                mixer.music.load(os.path.join('output', filename))
                mixer.music.play()
                self.current_playing = filename
                self.status_var.set(f"Playing: {filename}")
                self.equalizer_running = True
                self.animate_equalizer(self.eq_label)
                
                for i, (_, filename_, _) in enumerate(self.beats[:2]):
                    if filename_ == filename:
                        self.beat_widgets[i][0].config(highlightbackground="#A3BE8C")
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
        
        threading.Thread(target=play, daemon=True).start()
    
    def stop_beat(self):
        mixer.music.stop()
        self.current_playing = None
        self.status_var.set("Playback stopped")
        self.equalizer_running = False

        for widget in self.beat_widgets:
            widget[0].config(highlightbackground=self.accent_color)

def main(user_genre=None):
    global all_beats, features_for_all

    if not user_genre:
        user_genre = input("🎵 Enter a genre (classical, rock, jazz, hiphop): ").strip().lower()
    else:
        user_genre = user_genre.strip().lower()
    if user_genre not in genre_scales:
        print("❌ Invalid genre. Choose from classical, rock, jazz, hiphop.")
        return

    all_beats, features_for_all = generate_initial_population(user_genre, num_beats=500)
    
    for generation in range(150):
        print(f"\n=== Generation {generation + 1} ===")
        
        parents = select_best(all_beats, user_genre)  # Still selects 2 parents for evolution
        
        children, child_features = crossover(parents, user_genre, all_beats, features_for_all)
        children = update_scores(children, user_genre)

        mutated = mutate_beats(children, features_for_all, user_genre, num_mutations=3)
        
        combined_population = parents + children + mutated
        
        all_beats = sorted(combined_population, key=lambda x: x[0], reverse=True)[:500]
        features_for_all = [beat[2] for beat in all_beats]
        
        print(f"Best score this generation: {all_beats[0][0]}")

    final_top = select_final_best(all_beats, user_genre)  # Now selects only the single best
    
    root = tk.Tk()
    
    try:
        from ttkthemes import ThemedStyle
        style = ThemedStyle(root)
        style.set_theme("equilux")
    except:
        pass
    
    app = MusicPlayerGUI(root, final_top, user_genre)
    
    # Center window
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    GenreSelection(root)
    root.mainloop()