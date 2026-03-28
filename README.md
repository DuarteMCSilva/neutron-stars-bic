# BIC - Tidal Deformation of Neutron stars

Note: This work was done between 09/2019 and 08/2020, as part of an **undergraduate Research Grant**, under the supervision of Dr. [Márcio Ferreira](https://cfisuc.fis.uc.pt/people.php?oid=175621) and Dr. [Constança Providência](https://cfisuc.fis.uc.pt/people.php?oid=5038490), both distinguished figures in the field of neutron star astrophysics.

This research seeks to uncover the internal composition of neutron stars by bridging the gap between theoretical models and astronomical observations. By generating a vast dataset of potential **Equations of State** (EoS), one can derive the macroscopic properties (such as mass and radius) by solving the **Tolman-Oppenheimer-Volkoff** (TOV) equations.

Because neutron stars subject matter to extreme pressures and temperatures unattainable on Earth, they serve as unique cosmic laboratories. Comparing my calculated models against real-world observational data allows us to constrain the behavior of dense matter and search for evidence of exotic particles or phase transitions.


<img width="946" height="342" alt="image" src="https://github.com/user-attachments/assets/16664d8d-7ac4-4298-aee7-886f25afc735" />

Source: Astromaterial Science and Nuclear Pasta, M. E. Caplan, C. J. Horowitz

# Outputs

## Binary systems of neutron stars - Machine Learning approach

### Tidal deformation:
<p align="center">
    <img alt="image" src="output/machine_learning/L_sym(K_sym).png"/>
</p>

### Radius, Chirp Mass and Tidal Deformation
The $M_{chirp}$ of a two-body system can be expressed as:  $M_{chirp} = \frac{(m_1 m_2)^{3/5}}{(m_1 + m_2)^{1/5}}$

<p align="center">
    <img alt="image" src="output/machine_learning/Radius,M_chirp,Lambda.png"/>
</p>

## Solving TOV Equation from a Neutron Star EoS

Mass-radius relation and lambda-mass relation of a neutron star
<p align="center">
    <img src="output/tov_calculations/M(R)_and_Lambda(M).png"/>
</p>


## Running locally

### Environment

- Create python virtual environment:  `python -m venv .venv`

- Activate environment: `.venv\Scripts\activate.bat`

- Install requirements: `pip install -r requirements.txt`
