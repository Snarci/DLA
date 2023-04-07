

<style>
  jacopo {
    color: #F90;
  }
  snarci {
    color: #2F0;
  }
</style>

# Perché
File utile per ricordarci le informazioni più importanti

<snarci>Roba fatta da Snarci</snarci>

<jacopo>Roba fatta da Jacopo</jacopo>

<br>

# 1) CLASSIFICATION TASK SU CELLULE (CHAOYANG)
<!-- <details> -->


<snarci>

## 1a) Addestramento di un VIT NON Pre-trainato
<ul>
    <li><b>path del .py/.ipynb che genera il modello: </b>TODO 'ndo cazzo sta?</li>
    <li><b>path del .pth: </b>TODO 'ndo cazzo sta?</li>
    <li><b>Pre-processing da fare prima della prediction: </b>Ridimensione delle immagini? Dobbiamo usare un AutoImageProcessor? Se sì, quale?</li>
</ul>


## 1b) Fine-tuning di un VIT Pre-trainato
<ul>
    <li><b>path del .py/.ipynb che genera il modello: </b>TODO 'ndo cazzo sta?</li>
    <li><b>path del .pth: </b>TODO 'ndo cazzo sta?</li>
    <li><b>Pre-processing da fare prima della prediction: </b>Ridimensione delle immagini? Dobbiamo usare un AutoImageProcessor? Se sì, quale?</li>
</ul>
</snarci>

<jacopo>

## 1c) VIT distillation
<ul>
    <li><b>path del .py/.ipynb che genera il modello: </b>./distillation_test.ipynb</li>
    <li><b>path del .pth: </b>./jacoExperiments/best_distilled_model.pth</li>
    <li><b>Pre-processing da fare prima della prediction: </b>
        <ol>
            <li>conversione a 256<sup>2</sup> x 3</li>
            <li>conversione a tensore
        </ol>
    </li>
</ul>
<!-- </details> -->
</jacopo>

<br>

# 2) MASK FILLING TASK SU...

<snarci>

## 2a) UCCELLI
<ul>
    <li><b>path del .py/.ipynb che genera il modello: </b>TODO 'ndo cazzo sta?</li>
    <li><b>path del .pth: </b>TODO 'ndo cazzo sta?</li>
    <li><b>Pre-processing da fare prima della prediction: </b>Ridimensione delle immagini? Dobbiamo usare un AutoImageProcessor? Se sì, quale?</li>
</ul>
<snarci>

<jacopo>

## 2b) MELANZANE
<ul>
    <li><b>path del .py/.ipynb che genera il modello: </b>./mae_train su_melanzane.ipynb</li>
    <li><b>path del .pth: </b>./models/melanzana/_(call_save_pretrained)_BESTmodel.pth</li>
    <li><b>Pre-processing da fare prima della prediction: </b>
        <ol>
            <li>conversione a 224<sup>2</sup> x 3</li>
            <li>conversione a tensore</li>
            <li>Applicare l'<code>image_processor =
            AutoImageProcessor.from_pretrained("facebook/vit-mae-base") </code></li>
        </ol>
    </li>
</ul>
</jacopo>




