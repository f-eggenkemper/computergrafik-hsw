import cv2 # <- einfacher Import
import numpy as np # <- Umbenannter Import

print('Hallo Welt') # <- auf die Konsole schreiben

# Methode deklarieren
def methode1(name):
  print(name)

# Iterieren über Array
vornamen = ['Jana', 'Tim', 'Martin'] 
print(len(vornamen)) # <- Länge ausgeben per len()-Methode 
for einzelwert in vornamen: 
  print(einzelwert)
print("nach der for-Schleife")

# Iterieren über Range
for i in range(5):
  methode1(i) # <- Methodenaufruf

# If-Bedingung
for i in range(5):
  if i % 2 == 0:
    print(i)

