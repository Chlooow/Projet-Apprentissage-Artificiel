# Projet-Apprentissage-Artificiel
Dans le cadre du cours Apprentissage Artificiel 2025-2026

lien github : https://github.com/Chlooow/Projet-Apprentissage-Artificiel

## Informations
- **Chef de projet etudiant:** Chloé Nsonde Makoundou 
- **Manager** : Madame Rakia Jaziri
- **Formation** : Master 1 IBD - Université Paris 8 Saint-Denis-Vincennes
- **type de projet**: individuel

## Problématique

*Comment modéliser et prédire efficacement les prix des produits Dior en Chine, et quels sont les facteurs déterminants qui influencent le pricing dans le secteur du luxe ?*

Cette problématique implique deux axes principaux :

- Comprendre la structure des prix et les segments produits.
- Prédire le prix d’un produit à partir de ses caractéristiques.

## Dataset
le dataset est importé directement via huggingface
lien du dataset : https://huggingface.co/datasets/DBQ/Dior.Product.prices.China

Sinon il existe sur Databoutique.com :
https://www.databoutique.com/buy-data-page/dior+product-prices+china/r/recjNQrDONn090TZG

## Structure du projet
- /data : censé contenir le dataset. Il y a un sample_file.csv pour visualisé un peut du dataset
- /notebooks : Les notebooks pour travailler, voir les resultats plots...etc.
- /rapport : contient le rapport de présentation + les /figures téléchargées
- /src : contient nos fonctions auxiliaires
    - bonus.py : continent les fonctions aux bonus pour l'approche non sup + sub 
    - evaluate.py contient les fonction aux pour évaluer les models et les fonctions de plot
    - load_data.py : permet de prendre le dataset direct de huggingface
    - modelisation : contient les fonctions aux pour la modelisation des modeles
    - preprocessing.py : contient les fonction pour nettoyer les données
    - utils_eda.py: contient les fonction pour l'analyse exploratoire
- /.gitignore : permet d'ignorer des choses lorsqu'on met notre travail sur le git
- app.py : permet de faire l'interface 
- main.py : notre fichier de code principale
- README.md : fichier d'infos
- requirements.txt : les bibliotheques dépendances que j'ai téléchargé (mais je n'ai pas utilsé d'environnement virtuelle pour cette fois-ci)
- Proposition de sujet validée en .PDF
- LICENSE : la licence j'ai mis MIT

## Comment executer ?

```git clone https://github.com/Chlooow/Projet-Apprentissage-Artificiel```

```cd Projet-Apprentissage-Artificiel```

``` pip install -r requirements.txt ```

``` python3 main.py ou python main.py ```

## Rendu

Vendredi 5/12/25 00h

## Soutenance 

Mercredi 10/12/25

## Outils & Technologies utilisés

- Langage : Python, un tout petit peut de SQL pour voir les données sur Huggingface

- Kernel : Jupyter

- Cours/TP madame Jaziri, Cours/TP de François Landes (L3), Machine Learning (Apprentissage Supervisée et Non Supervisé)

## Autrice 

Makoundou Chloé

Ce projet illustre ma capacité à mener un projet de machine learning de bout en bout à partir des connaissances que j'ai eu en Master 1, de l’analyse exploratoire à l’interprétation des résultats, en gardant un souci d’applicabilité réelle dans le secteur du luxe. Ceci m'aide à me rappeler et pratiqué les notions déjà utilisé.

