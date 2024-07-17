# %%
import cartopy.crs as ccrs
import logging
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

logging.basicConfig(level=logging.INFO)

# %%
tech_colors = {"gas": "#FFCD96", "wind": "#417505"}

# %%
# Meshed AC DC example
# DC lines between UK, Germany and Norway (red)
# Potential link between London and Bremen (green)
network = pypsa.examples.ac_dc_meshed(from_master=True)
lines_current_type = network.lines.bus0.map(network.buses.carrier)
network.plot(
    line_colors=lines_current_type.map(
        lambda ct: "indianred" if ct == "AC" else "seagreen"
    ),
    title="Example network (AC is red, DC is green)",
    color_geomap=True,
    jitter=0.3,
)
plt.show()


# %%
# Optimize network. optimize method mimics lopf
network.optimize(solver_name="highs")

# plot generation and line flows
generation = network.generators_t.p.sum()
generators = network.generators.copy()
generators["gen"] = network.generators_t.p.mean()
generation_by_bus_carrier = generators.groupby(["bus", "carrier"]).gen.sum()
collection = network.plot(
    bus_sizes=generation_by_bus_carrier / 5e3,
    bus_colors=tech_colors,
    margin=0.5,
    flow="mean",
    line_widths=0.2,
    link_widths=0.0,
    color_geomap=True,
    projection=ccrs.EqualEarth(),
    line_colors=network.lines_t.p0.mean().abs(),
)
plt.colorbar(collection[2], fraction=0.04, pad=0.004, label="Flow in MW")
pypsa.plot.add_legend_patches(
    plt.gca(), colors=list(tech_colors.values()), labels=list(tech_colors.keys())
)

# %%
# Optimize network using rolling horizon
network_roll = pypsa.examples.ac_dc_meshed(from_master=True)
pypsa.optimization.optimize.optimize_with_rolling_horizon(
    network_roll, network_roll.snapshots, 2, 1, solver_name="highs"
)

# plot generation and line flows
generation = network_roll.generators_t.p.sum()
generators = network_roll.generators.copy()
generators["gen"] = network_roll.generators_t.p.mean()
generation_by_bus_carrier = generators.groupby(["bus", "carrier"]).gen.sum()
collection = network_roll.plot(
    bus_sizes=generation_by_bus_carrier / 5e3,
    bus_colors=tech_colors,
    margin=0.5,
    flow="mean",
    line_widths=0.2,
    link_widths=0.0,
    color_geomap=True,
    projection=ccrs.EqualEarth(),
    line_colors=network_roll.lines_t.p0.mean().abs(),
)
plt.colorbar(collection[2], fraction=0.04, pad=0.004, label="Flow in MW")
pypsa.plot.add_legend_patches(
    plt.gca(), colors=list(tech_colors.values()), labels=list(tech_colors.keys())
)

# %% export network data
network.export_to_netcdf("data/acdc_network.nc")