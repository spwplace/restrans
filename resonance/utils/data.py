"""
Data loading utilities for Resonance Transformer experiments.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import random


class TinyShakespeare(Dataset):
    """Character-level dataset for quick experiments."""

    def __init__(self, text: str, seq_len: int = 128):
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return max(1, len(self.text) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.text[start:end]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + ' ' * (self.seq_len + 1 - len(chunk))
        tokens = [self.stoi[ch] for ch in chunk]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


class SyntheticTokenDataset(Dataset):
    """Synthetic token sequences with local structure."""

    def __init__(self, vocab_size: int = 1000, seq_len: int = 128, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate sequences with Markov-like local dependencies
        x = torch.zeros(self.seq_len, dtype=torch.long)
        x[0] = random.randint(0, self.vocab_size - 1)
        for t in range(1, self.seq_len):
            # 80% probability of staying near previous token, 20% random
            if random.random() < 0.8:
                x[t] = (x[t-1] + random.randint(-5, 5)) % self.vocab_size
            else:
                x[t] = random.randint(0, self.vocab_size - 1)
        y = torch.roll(x, shifts=-1, dims=0)
        y[-1] = random.randint(0, self.vocab_size - 1)
        return x, y


def get_shakespeare_text():
    """Return tiny shakespeare text or generate synthetic."""
    text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take Arms against a Sea of troubles,
And by opposing end them: to die, to sleep;
No more; and by a sleep, to say we end
The heart-ache, and the thousand natural shocks
That Flesh is heir to? 'Tis a consummation
Devoutly to be wished. To die, to sleep,
To sleep, perchance to Dream; aye, there's the rub,
For in that sleep of death, what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes Calamity of so long life:
For who would bear the Whips and Scorns of time,
The Oppressor's wrong, the proud man's Contumely,
The pangs of dispised Love, the Law's delay,
The insolence of Office, and the spurns
That patient merit of the unworthy takes,
When he himself might his Quietus make
With a bare Bodkin? Who would Fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovered country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have,
Than fly to others that we know not of.
Thus conscience does make cowards of us all,
And thus the native hue of Resolution
Is sicklied o'er, with the pale cast of Thought,
And enterprises of great pitch and moment,
With this regard their Currents turn awry,
And lose the name of Action.
"""
    return text * 10  # Repeat for more data


def get_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
