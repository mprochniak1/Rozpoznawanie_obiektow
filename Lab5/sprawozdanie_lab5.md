# Sprawozdanie laboratoryjne

**Przedmiot:** Uczenie głębokie do rozpoznawania obiektów
**Zajęcia nr / temat:** 5 / Warstwy splotowe - operacja, parametry, interpretacja
**Autor / grupa:** Maciej Próchniak / Gr.3
**Data:** 18.11.2026

## 1. Cel ćwiczenia

Celem ćwiczenia było zbadanie działania warstwy splotowej na jednym zdjęciu w skali szarości. Sprawdziłem, co robią ręcznie dobrane jądra, jak rozmiar jądra, stride, padding i dilation wpływają na rozmiar wyjścia, na co reagują filtry pierwszej warstwy pretrenowanej sieci oraz czym różnią się filtry rozmywające Box i Gaussian.

## 2. Metoda i implementacja

![Zdjęcie w skali szarości z zaznaczonym fragmentem](zdjecie_szarosc.png)

**Zdjęcie.** Półka z książkami, 1500 × 867 pikseli, zamieniona na skalę szarości (wartości 0-1). Do badania parametrów splotu i porównania rozmyć użyto fragmentu 210 × 320 pikseli zaznaczonego czerwoną ramką, z napisami „Writing on the Wall” i „CHRISTIAN JENNINGS”.

**Narzędzia.** PyTorch 2.14.1, torchvision, matplotlib, pandas i Pillow.

**Konfiguracja.** Jądra 3 × 3 (Sobel X i Y, Laplacian, wyostrzanie) zastosowano przez F.conv2d z padding=1. Rozmiar wyjścia liczono ze wzoru out = (in + 2p - d(k - 1) - 1) / s + 1, zaokrąglonego w dół, i porównano z nn.Conv2d. Mapy cech pochodzą z warstwy conv1 pretrenowanego ResNet-18 (zdjęcie 224 × 388, ta sama szarość w trzech kanałach). Filtry Box i Gaussian zastosowano ze stride=1, dilation=1 i padding równym k // 2, a fragment pokazano na wspólnej skali jasności 0-1.

**Kod.** Cały kod znajduje się w pliku lab5_Maciej_Próchniak.ipynb w folderze Lab5 repozytorium: https://github.com/mprochniak1/Rozpoznawanie_obiektow/tree/main/Lab5

## 3. Wyniki

![Wyniki ręcznych jąder](reczne_jadra.png)

Sobel X zaznacza pionowe krawędzie grzbietów, a Sobel Y poziome krawędzie i napisy na grzbietach. Magnituda pokazuje kontury wszystkich książek i liter, Laplacian cienkie linie wzdłuż krawędzi, a po wyostrzaniu napisy są ostrzejsze.

| k | stride | padding | dilation | Wzór | nn.Conv2d |
|---:|---:|---:|---:|---|---|
| 3 | 1 | 0 | 1 | 865 × 1498 | 865 × 1498 |
| 3 | 1 | 1 | 1 | 867 × 1500 | 867 × 1500 |
| 5 | 2 | 2 | 1 | 434 × 750 | 434 × 750 |
| 3 | 1 | 2 | 2 | 867 × 1500 | 867 × 1500 |
| 7 | 3 | 0 | 1 | 287 × 498 | 287 × 498 |
| 4 | 2 | 1 | 3 | 430 × 747 | 430 × 747 |

Dla wejścia 867 × 1500 wynik wzoru zgadza się z nn.Conv2d we wszystkich przypadkach.

![Wpływ stride, padding i dilation](stride_padding_dilation.png)

Fragment 210 × 320 daje mapę 210 × 320 przy padding=1, 208 × 318 przy padding=0, 214 × 324 przy padding=3, 105 × 160 przy stride=2 i 210 × 320 przy dilation=2 z padding=2.

![Filtry warstwy conv1](filtry_conv1.png)

![Mapy cech warstwy conv1](mapy_cech_conv1.png)

Najsilniejsze odpowiedzi dały mapy 26 (maksimum 35,9) i 14 (34,8), a dziesięć map ma maksimum 0,0.

![Porównanie filtrów rozmywających](filtry_rozmywajace.png)

Wszystkie trzy filtry zachowały rozmiar 867 × 1500. Gaussian 5 × 5 ma wagi od 36/256 w środku do 1/256 w rogach.

## 4. Analiza i interpretacja

**Ręczne jądra.** Sobel X reaguje na pionowe krawędzie, a Sobel Y na poziome, dlatego obrócone napisy na grzbietach lepiej widać w Sobelu Y. Laplacian ma sumę wag równą zero, więc reaguje tylko na zmiany jasności, a wyostrzanie zwiększa lokalny kontrast.

**Stride, padding i dilation.** Brak dopełnienia zmniejsza mapę, a padding=3 ją powiększa i daje jasne linie na brzegach, gdzie Sobel X widzi przejście od zer do zdjęcia. Stride=2 zmniejsza mapę o połowę, a dilation=2 poszerza okno do 5 × 5 przy 9 wagach, przez co krawędzie są szersze.

**Mapy cech.** Filtry conv1 działają jak detektory pionowych i poziomych krawędzi (np. 14, 26), tekstur i napisów (np. 39, 63) oraz jasnych lub ciemnych obszarów. Filtry kolorowe (np. 5, 28) prawie nie reagują, bo zdjęcie ma we wszystkich kanałach tę samą wartość, a mapy z maksimum 0,0 pochodzą głównie od filtrów o wagach bliskich zeru.

**Box i Gaussian.** Box 5 × 5 rozmywa mocniej niż Box 3 × 3, bo uśrednia 25 pikseli zamiast 9, przez co cienkie litery „Writing on the Wall” się zlewają. Gaussian 5 × 5 rozmywa słabiej niż Box 5 × 5 i te same litery pozostają czytelne, bo ma ok. 77% wagi w środkowym obszarze 3 × 3, a Box 5 × 5 tylko 36%.

## 5. Wnioski

Ręczne jądra działają zgodnie ze swoimi wagami: Sobel X i Y wykrywają pionowe i poziome krawędzie, Laplacian zmiany jasności, a wyostrzanie zwiększa kontrast. Wzór na rozmiar wyjścia zgadza się z nn.Conv2d we wszystkich sześciu przypadkach, a stride zmniejsza mapę, padding ją zachowuje lub powiększa, a dilation poszerza zasięg okna bez dodawania wag. Filtry conv1 ResNet-18 przypominają detektory krawędzi i tekstur, a Gaussian 5 × 5 rozmywa słabiej niż Box 5 × 5, prawdopodobnie dzięki skupieniu wag w środku okna.
