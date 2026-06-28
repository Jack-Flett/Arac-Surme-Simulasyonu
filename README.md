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

- Python 3.11 (daha üst sürümlerde paket versiyon sıkıntıları çıkıyor)
- pip

### Python 3.11 İndirme
 
Python 3.11 yüklü değilse: https://www.python.org/downloads/release/python-31115/
 
Kurulum sırasında **"Add python.exe to PATH"** seçeneğini işaretlemenize gerek yok.

Kurulduğunu görmek için:
```powershell
py -3.11 --version
```

Veya
```powershell
py --list
```

### Adımlar

1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/Jack-Flett/Arac-Surme-Simulasyonu.git
   cd Arac-Surme-Simulasyonu
   ```

2. Python 3.11 ile sanal ortam oluşturun:
    ```powershell
    py -3.11 -m venv venv
    ```
 
3. Sanal ortamı etkinleştirin:
    ```powershell
    venv\Scripts\Activate.ps1
    ```
   > Hata alırsanız önce şunu çalıştırın:
   > ```powershell
   > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   > ```
   > Sonra tekrar deneyin.
 
4. Bağımlılıkları yükleyin:
    ```powershell
    pip install -r requirements.txt
    ```
 
5. Projeyi çalıştırın:
    ```powershell
    python main.py
    ```
 
### Sanal Ortamdan Çıkış
 
 Sanal ortamı kapatmak için:
```powershell
deactivate
```
 
### Sonraki Çalıştırmalarda
 
Sanal ortamı tekrar etkinleştirip çalıştırmanız yeterlidir:
```powershell
venv\Scripts\Activate.ps1
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