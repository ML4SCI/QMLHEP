# Learning quantum representations of classical high energy physics data with contrastive learning

<p>
<!-- [<img src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/gsoc%40ml4sci.jpeg" title="Electron" />](https://ml4sci.org/) -->
<a href="https://ml4sci.org/" target="_blank"><img alt="gsoc@ml4sci" height="350px" width="1000" src="https://raw.githubusercontent.com/Amey-2002/GSoC_2022_QMLHEP/main/assets/gsoc%40ml4sci.jpeg" /></a>
</p>

This project is an official submission to the [Google Summer of Code 2024](https://summerofcode.withgoogle.com/) program carried out under the supervision of mentors from the ML4SCI organization.<br>
The official project webpage can be found [here](https://summerofcode.withgoogle.com/programs/2024/projects/nyIWIGtE).

#### A comprehensive description of the project can be found in these two blog posts -

[Quantum Contrastive Learning for High Energy Physics](https://medium.com/@ameybhatuse315/quantum-contrastive-learning-for-high-energy-physics-2d8fb7368708)

[Quantum Graph Contrastive Learning for High Energy Physics](https://medium.com/@ameybhatuse315/quantum-graph-contrastive-learning-for-high-energy-physics-aa6e49eaa34f)


<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;  margin-right: 30px;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-cly1{text-align:left;vertical-align:middle}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style>

<div>
<table class="tg" style="undefined;table-layout: fixed; width: 292px;float : left">
<caption style="text-align:center;"><font size='5'><b>Accuracy</b></font></caption>
<colgroup>
<col style="width: 109.2px">
<col style="width: 91.2px">
<col style="width: 91.2px">
</colgroup>
<thead>
  <tr>
    <th class="tg-amwm">Particle Count</th>
    <th class="tg-amwm">Classical</th>
    <th class="tg-amwm">Quantum</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-baqh">5</td>
    <td class="tg-cly1">67.62 ± 1.11</td>
    <td class="tg-cly1">68.46 ± 1.28</td>
  </tr>
  <tr>
    <td class="tg-baqh">6</td>
    <td class="tg-cly1">71.96 ± 1.25</td>
    <td class="tg-cly1">70.18 ± 1.32</td>
  </tr>
  <tr>
    <td class="tg-baqh">7</td>
    <td class="tg-cly1">66.86 ± 1.07</td>
    <td class="tg-cly1">68.14 ± 1.25</td>
  </tr>
  <tr>
    <td class="tg-baqh">8</td>
    <td class="tg-cly1">70.86 ± 0.64</td>
    <td class="tg-cly1">70.24 ± 0.9</td>
  </tr>
  <tr>
    <td class="tg-baqh">9</td>
    <td class="tg-cly1">66.88 ± 1.37</td>
    <td class="tg-cly1">67.28 ± 1.76</td>
  </tr>
  <tr>
    <td class="tg-baqh">10</td>
    <td class="tg-cly1">69.34 ± 1.13</td>
    <td class="tg-cly1">69.84 ± 1.07</td>
  </tr>
</tbody>
</table>
<table class="tg" style="undefined;table-layout: fixed; width: 292px">
<caption style="text-align:center;"><font size='5'><b>AUC</b></font></caption>
<colgroup>
<col style="width: 109.2px">
<col style="width: 91.2px">
<col style="width: 91.2px">
</colgroup>
<thead>
  <tr>
    <th class="tg-amwm">Particle Count</th>
    <th class="tg-amwm">Classical</th>
    <th class="tg-amwm">Quantum</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-baqh">5</td>
    <td class="tg-cly1">67.62 ± 1.11</td>
    <td class="tg-cly1">68.46 ± 1.28</td>
  </tr>
  <tr>
    <td class="tg-baqh">6</td>
    <td class="tg-cly1">71.96 ± 1.25</td>
    <td class="tg-cly1">70.18 ± 1.32</td>
  </tr>
  <tr>
    <td class="tg-baqh">7</td>
    <td class="tg-cly1">66.86 ± 1.07</td>
    <td class="tg-cly1">68.14 ± 1.25</td>
  </tr>
  <tr>
    <td class="tg-baqh">8</td>
    <td class="tg-cly1">70.86 ± 0.64</td>
    <td class="tg-cly1">70.24 ± 0.9</td>
  </tr>
  <tr>
    <td class="tg-baqh">9</td>
    <td class="tg-cly1">66.88 ± 1.37</td>
    <td class="tg-cly1">67.28 ± 1.76</td>
  </tr>
  <tr>
    <td class="tg-baqh">10</td>
    <td class="tg-cly1">69.34 ± 1.13</td>
    <td class="tg-cly1">69.84 ± 1.07</td>
  </tr>
</tbody>
</table>
</div>
<table class="tg" style="undefined;table-layout: fixed; width: 292px;margin-left:auto;margin-right:auto">
<caption style="text-align:center;"><font size='5'><b>F1-Score</b></font></caption>
<colgroup>
<col style="width: 109.2px">
<col style="width: 91.2px">
<col style="width: 91.2px">
</colgroup>
<thead>
  <tr>
    <th class="tg-amwm">Particle Count</th>
    <th class="tg-amwm">Classical</th>
    <th class="tg-amwm">Quantum</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-baqh">5</td>
    <td class="tg-cly1">67.62 ± 1.11</td>
    <td class="tg-cly1">68.46 ± 1.28</td>
  </tr>
  <tr>
    <td class="tg-baqh">6</td>
    <td class="tg-cly1">71.96 ± 1.25</td>
    <td class="tg-cly1">70.18 ± 1.32</td>
  </tr>
  <tr>
    <td class="tg-baqh">7</td>
    <td class="tg-cly1">66.86 ± 1.07</td>
    <td class="tg-cly1">68.14 ± 1.25</td>
  </tr>
  <tr>
    <td class="tg-baqh">8</td>
    <td class="tg-cly1">70.86 ± 0.64</td>
    <td class="tg-cly1">70.24 ± 0.9</td>
  </tr>
  <tr>
    <td class="tg-baqh">9</td>
    <td class="tg-cly1">66.88 ± 1.37</td>
    <td class="tg-cly1">67.28 ± 1.76</td>
  </tr>
  <tr>
    <td class="tg-baqh">10</td>
    <td class="tg-cly1">69.34 ± 1.13</td>
    <td class="tg-cly1">69.84 ± 1.07</td>
  </tr>
</tbody>
</table>

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style>
<div>
<table class="tg" style="undefined;table-layout: fixed; width: 333px;float : left">
<caption style="text-align:center;"><font size='5'><b>Accuracy</b></font></caption>
<colgroup>
<col style="width: 150.2px">
<col style="width: 91.2px">
<col style="width: 91.2px">
</colgroup>
<thead>
  <tr>
    <th class="tg-7btt">Augmentation Ratio</th>
    <th class="tg-7btt">Classical</th>
    <th class="tg-7btt">Quantum</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-lboi">68.98 ± 1.46</td>
    <td class="tg-lboi">68.8 ± 1.01</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.3</td>
    <td class="tg-lboi">68.62 ± 1.03</td>
    <td class="tg-lboi">69.54 ± 1.92</td>
  </tr>
</tbody>
</table>
<table class="tg" style="undefined;table-layout: fixed; width: 333px">
<caption style="text-align:center;"><font size='5'><b>AUC</b></font></caption>
<colgroup>
<col style="width: 150.2px">
<col style="width: 91.2px">
<col style="width: 91.2px">
</colgroup>
<thead>
  <tr>
    <th class="tg-7btt">Augmentation Ratio</th>
    <th class="tg-7btt">Classical</th>
    <th class="tg-7btt">Quantum</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-lboi">0.753 ± 0.01</td>
    <td class="tg-lboi">0.752 ± 0.01</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.3</td>
    <td class="tg-lboi">0.746 ± 0.02</td>
    <td class="tg-lboi">0.759 ± 0.02</td>
  </tr>
</tbody>
</table>
</div>
<table class="tg" style="undefined;table-layout: fixed; width: 333px;margin-left:auto;margin-right:auto">
<caption style="text-align:center;"><font size='5'><b>F1-Score</b></font></caption>
<colgroup>
<col style="width: 150.2px">
<col style="width: 91.2px">
<col style="width: 91.2px">
</colgroup>
<thead>
  <tr>
    <th class="tg-7btt">Augmentation Ratio</th>
    <th class="tg-7btt">Classical</th>
    <th class="tg-7btt">Quantum</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-c3ow">0.2</td>
    <td class="tg-lboi">0.69 ± 0.01</td>
    <td class="tg-lboi">0.688 ± 0.01</td>
  </tr>
  <tr>
    <td class="tg-c3ow">0.3</td>
    <td class="tg-lboi">0.686 ± 0.01</td>
    <td class="tg-lboi">0.695 ± 0.02</td>
  </tr>
</tbody>
</table> -->