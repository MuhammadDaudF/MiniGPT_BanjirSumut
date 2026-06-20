TinyGPT - BanjirSumut

Implementasi mini-GPT dari nol menggunakan PyTorch, sebagai tugas mata kuliah Proyek Data Mining (ST167). Project ini melatih model transformer sederhana dari corpus teks tentang banjir bandang Sumatera Utara, dengan membandingkan empat pendekatan tokenisasi: character, word, SentencePiece BPE, dan Tiktoken.

Struktur File

FileFungsicorpus.txtData teks training (±7.000 kata, domain: banjir bandang Sumut)transformer_blocks.pyModul arsitektur transformer (Self-Attention, Multi-Head Attention, Feed Forward, Block) — dipakai bersama oleh semua file trainingtinygpt_char.pyTraining model dengan tokenisasi per karaktertinygpt_word.pyTraining model dengan tokenisasi per katatinygpt_bpe.pyTraining model dengan tokenisasi SentencePiece BPE (dilatih dari corpus sendiri)tinygpt_tiktoken.pyTraining model dengan tokenizer Tiktoken (vocab pretrained dari OpenAI)run_benchmark.pyOrchestrator — menjalankan keempat metode di atas secara otomatis berurutan dan menyimpan hasil benchmark

Instalasi

Pastikan Python sudah terpasang, lalu install dependency yang dibutuhkan:

bashpy -m pip install torch sentencepiece tiktoken

Cara Eksekusi

Opsi A — Jalankan satu per satu

Setiap file training bisa dijalankan independen, tidak ada urutan wajib, asal corpus.txt dan transformer_blocks.py ada di folder yang sama:

bashpy tinygpt_char.py
py tinygpt_word.py
py tinygpt_bpe.py
py tinygpt_tiktoken.py

Masing-masing akan mencetak ke terminal: ukuran vocabulary, jumlah token hasil encoding, progres loss tiap 300 step, loss akhir, dan contoh teks hasil generate.

Opsi B — Jalankan semua otomatis sekaligus (rekomendasi untuk benchmark)

bashpy run_benchmark.py

Script ini akan menjalankan keempat metode tokenisasi secara berurutan dalam satu proses, lalu menyimpan hasilnya ke dua file:


benchmark_results.json — hasil lengkap (vocab size, jumlah parameter, waktu training, riwayat loss, teks hasil generate)
benchmark_results.csv — ringkasan tabel, siap dibuka di Excel untuk dilampirkan ke laporan


Catatan


Training berjalan di CPU jika tidak ada GPU/CUDA tersedia. Untuk Tiktoken, waktu training lebih lambat karena vocab size jauh lebih besar (~100.000 token) dibanding tiga metode lainnya.
Hyperparameter (embedding_dim, n_heads, n_layers, learning_rate, epochs) dibuat identik di semua metode agar perbandingan antar tokenizer adil — hanya bagian tokenizer yang berbeda.
File tokenizer_bpe.model dan tokenizer_bpe.vocab akan otomatis dibuat ulang setiap kali tinygpt_bpe.py dijalankan (hasil training SentencePiece dari corpus.txt).
