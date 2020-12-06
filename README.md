# Formation HPC Scalian

## Procédure de compilation
**Build d'une solution via qmake**

Testé uniquement pour Visual Studio 2017

**Lancer la solution**

Renseigner le fichier « build_and_run_solution.bat »

Le chemin vers CUDA doit être un « short path » : 
ouvrir une fenêtre powershell à la racine du dépôt et taper la commande suivante :.\getShortPath.cmd "Path complet  vers CUDA"

La commande vous renvoie le chemin court

Le champ WIN_SDK se trouve en faisant un clic droit propriété sur un projet Visual, onglet Général

## Contenu

Support pour la formation HPC de Scalian DS

Contient les diapos et les exercices concernant chacun des thèmes abordés :
- SIMD
- OpenMP
- CUDA