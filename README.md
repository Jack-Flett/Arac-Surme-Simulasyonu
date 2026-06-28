# Araç Sürme Simülasyonu

OpenGL tabanlı, gerçek zamanlı bir araç sürme simülasyonu. PyOpenGL ve Pygame kullanılarak Python'da geliştirilmiştir. Oyuncu, gece atmosferinde aydınlatılmış bir kasabada araç sürebilir; serbest kamera veya takip kamerası ile dünyayı keşfedebilir.

---

## Kullanılan Kütüphaneler

| Kütüphane | Versiyon | Kullanım Amacı |
|-----------|----------|----------------|
| PyOpenGL | 3.1.7 | OpenGL 3.3 Core Profile rendering |
| PyOpenGL_accelerate | 3.1.7 | OpenGL performans optimizasyonu |
| pygame | 2.5.2 | Pencere yönetimi, klavye/fare girişi |
| numpy | 1.26.4 | Matris ve vektör hesaplamaları |
| Pillow | 10.3.0 | Texture yükleme |

---

## Kurulum ve Çalıştırma

### Gereksinimler

- Python 3.11 veya üzeri
- pip

### Adımlar

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/Jack-Flett/Arac-Surme-Simulasyonu.git
   cd Arac-Surme-Simulasyonu
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Projeyi çalıştırın:
   ```bash
   python main.py
   ```

### Klasör Yapısı

```
├── main.py
├── car.py
├── camera.py
├── renderer.py
├── scene.py
├── requirements.txt
├── shaders/
│   ├── vertex.glsl
│   └── fragment.glsl
├── textures/
│   ├── grass.png
│   ├── road.png
│   ├── pavement.png
│   └── building.png
└── models/
    └── car.obj
```

---

## Kontrol Tuşları

### Sürüş (Chase Modu)

| Tuş | Eylem |
|-----|-------|
| `W` | İleri git / hızlan |
| `S` | Geri git / fren |
| `A` | Sola dön |
| `D` | Sağa dön |
| `Fare` | Kamerayı döndür |

### Serbest Kamera (Free-Fly Modu)

| Tuş | Eylem |
|-----|-------|
| `W / S` | İleri / geri uç |
| `A / D` | Sola / sağa uç |
| `Space` | Yukarı çık |
| `Sol Shift` | Aşağı in |
| `Fare` | Bakış yönünü değiştir |

### Genel

| Tuş | Eylem |
|-----|-------|
| `F` | Serbest kamera ↔ Takip kamerası geçiş |
| `B` | Çarpışma kutularını göster / gizle (debug) |
| `Escape` | Çıkış |