# pyWOMBAT
=======
Here we provide 0D chemostat and 1D water column models of the Australian ocean biogeochemical model WOMBAT (World Ocean Model of Biogeochemistry and Trophic-dynamics).
It is written in python and intended for development purposes.

## Components
- run.py            [Master file]
- bgc_sms.py        [BGC routine]
- advect_diff.py    [Mixing and advection]
- phyparams.py      [Physical input parameters and z-grid]
- bgcparams.py      [Biogeochemical input parameters]
- tra_init.py       [Tracer initialisation]
- mass_balance.py   [Checks for mass balance conservation]

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


