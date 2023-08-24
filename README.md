# pyWOMBAT
=======
Here we provide 0D chemostat and 1D water column models of the Australian ocean biogeochemical model WOMBAT (World Ocean Model of Biogeochemistry and Trophic-dynamics).
It is written in python and intended for development purposes.

## 0D model components
- run_0D.py            [Master file]
- bgc_sms_0D.py        [BGC routine]
- mass_balance_0D.py   [Checks for mass balance conservation]

## 1D model components
- run.py            [Master file]
- bgc_sms.py        [BGC routine]
- advect_diff.py    [Mixing and advection]
- phyparams.py      [Physical input parameters and z-grid]
- bgcparams.py      [Biogeochemical input parameters]
- tra_init.py       [Tracer initialisation]
- mass_balance.py   [Checks for mass balance conservation]

## Structure of run_0D.py
1. Set timestep, run length and logicals
2. Initialise tracers
3. Loops through model                          [calls **bgc_sms_0D**]..
                                                [calls **mass_balance_0D**]
4. Final call of BGC routine                    [calls **bgc_sms_0D**]
5. Save restart to output file
6. Visualise the results

## Structure of run.py
1. Set timestep, run length and logicals
2. Loads physical parameters and vertical grid  [calls **phyparams**]
3. Initialise tracers                           [calls **tra_init**]
4. Loads biogeochemical parameters              [calls **bgcparams**]
5. Loops through model                          [calls **advect_diff**]..
                                                [calls **mix_mld**]..
                                                [calls **bgc_sms**]..
                                                [calls **massb_c**]..
                                                [calls **massb_n**]..
                                                [calls **massb_p**]..
                                                [calls **massb_fe**]
6. Final call of BGC routine                    [calls **bgc_sms**]
7. Save output to output file
8. Visualise the results


