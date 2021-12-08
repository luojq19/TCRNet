# README
- Current approach:
    - Network: CNN
    - Encoding: one-hot or Blosum50 matrix
- Possible alternative approach:
    - Network: RNN
    - Encoding: kmers + bag of word or kmers + word embedding
- A small problem: what does the 'X' in the sequences stands for? For now I will just assume it's a sign of unknown amino acid and encode it as all 0
