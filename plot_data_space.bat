@echo off

python plot_data_space.py ^
    mesh.txt ^
    results\vapor_velocity.txt ^
    results\vapor_pressure.txt ^
    results\vapor_temperature.txt ^
    results\rho_vapor.txt ^
    results\vapor_alpha.txt ^
    results\liquid_velocity.txt ^
    results\liquid_pressure.txt ^
    results\liquid_temperature.txt ^
    results\liquid_rho.txt ^
    results\liquid_alpha.txt ^
    results\wall_temperature.txt
pause
