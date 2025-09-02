#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KG booster: synthesize a small but rich augmentation to your domain ontology
so the dataset generators produce MANY recall (object facts) and logic (inferred types).

Writes (does NOT modify originals):
  backend/ontologies/<domain>_augment.ttl

You can then append it into your main TTL with `cat ... >> ...` (see steps below).
"""
from __future__ import annotations
import argparse, random
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
ONT  = ROOT / "backend" / "ontologies"

HDR = """@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix ext:  <http://example.org/ext#> .

"""

def lbl(u: str, s: str) -> str:
    s = s.replace('"','')
    return f'{u} rdfs:label "{s}" .\n'

def battery_block(n: int) -> str:
    out = []
    # classes + hierarchy
    out += [
        "ext:Product a owl:Class .",
        "ext:Battery a owl:Class ; rdfs:subClassOf ext:Product .",
        "ext:RechargeableBattery a owl:Class ; rdfs:subClassOf ext:Battery .",
        "ext:LithiumIonBattery a owl:Class ; rdfs:subClassOf ext:RechargeableBattery .",
        "ext:Spec a owl:Class .",
        "ext:Capacity a owl:Class ; rdfs:subClassOf ext:Spec .",
        "ext:Voltage a owl:Class ; rdfs:subClassOf ext:Spec .",
        "ext:Accessory a owl:Class ; rdfs:subClassOf ext:Product .",
        "ext:Manufacturer a owl:Class .",
        "ext:hasSpec a owl:ObjectProperty ; rdfs:domain ext:Product ; rdfs:range ext:Spec .",
        "ext:hasCapacity a owl:ObjectProperty ; rdfs:subPropertyOf ext:hasSpec ; rdfs:domain ext:Battery ; rdfs:range ext:Capacity .",
        "ext:hasVoltage a owl:ObjectProperty ; rdfs:subPropertyOf ext:hasSpec ; rdfs:domain ext:Battery ; rdfs:range ext:Voltage .",
        "ext:compatibleWith a owl:ObjectProperty ; rdfs:domain ext:Product ; rdfs:range ext:Accessory .",
        "ext:madeBy a owl:ObjectProperty ; rdfs:domain ext:Product ; rdfs:range ext:Manufacturer .",
        lbl("ext:Battery","Battery"),
        lbl("ext:RechargeableBattery","Rechargeable Battery"),
        lbl("ext:LithiumIonBattery","Lithium-ion Battery"),
        lbl("ext:Capacity","capacity"),
        lbl("ext:Voltage","voltage"),
        lbl("ext:Accessory","accessory"),
        lbl("ext:Manufacturer","manufacturer"),
        lbl("ext:hasCapacity","capacity"),
        lbl("ext:hasVoltage","voltage"),
        lbl("ext:compatibleWith","compatible accessory"),
        lbl("ext:madeBy","manufacturer"),
    ]
    # manufacturers + accessories
    for m in ["VoltMax","PowerCore","Energen","CellTech","NanoVolt"]:
        out += [f"ext:man_{m} a ext:Manufacturer .", lbl(f"ext:man_{m}", m)]
    for a in ["BMS-Guard","Case-Pro","QuickMount","FuseKit","ThermalPad"]:
        out += [f"ext:acc_{a} a ext:Accessory .", lbl(f"ext:acc_{a}", a)]
    # instances
    rnd_cap = [1800, 2200, 2500, 3000, 3500, 4000]
    rnd_v   = [3.7, 7.4, 12.0, 18.0, 24.0]
    for i in range(n):
        prod = f"ext:batt_{i:03d}"
        capv = random.choice(rnd_cap)
        volt = random.choice(rnd_v)
        man  = f"ext:man_{random.choice(['VoltMax','PowerCore','Energen','CellTech','NanoVolt'])}"
        acc  = f"ext:acc_{random.choice(['BMS-Guard','Case-Pro','QuickMount','FuseKit','ThermalPad'])}"
        cap  = f"ext:cap_{i:03d}"
        vol  = f"ext:vol_{i:03d}"
        out += [
            f"{prod} a ext:LithiumIonBattery .",
            lbl(prod, f"LiCell-{i:03d}"),
            f"{cap} a ext:Capacity .", lbl(cap, f"{capv} mAh"),
            f"{vol} a ext:Voltage .",  lbl(vol, f"{volt:g} V"),
            f"{prod} ext:hasCapacity {cap} .",
            f"{prod} ext:hasVoltage {vol} .",
            f"{prod} ext:madeBy {man} .",
            f"{prod} ext:compatibleWith {acc} .",
        ]
    return "\n".join(out) + "\n"

def lexmark_block(n: int) -> str:
    out = []
    out += [
        "ext:Printer a owl:Class .",
        "ext:LaserPrinter a owl:Class ; rdfs:subClassOf ext:Printer .",
        "ext:Cartridge a owl:Class .",
        "ext:Yield a owl:Class .",
        "ext:Accessory a owl:Class .",
        "ext:usesCartridge a owl:ObjectProperty ; rdfs:domain ext:Printer ; rdfs:range ext:Cartridge .",
        "ext:hasYield a owl:ObjectProperty ; rdfs:domain ext:Cartridge ; rdfs:range ext:Yield .",
        "ext:supportsDuplex a owl:ObjectProperty ; rdfs:domain ext:Printer ; rdfs:range ext:Accessory .",
        lbl("ext:Printer","printer"),
        lbl("ext:LaserPrinter","laser printer"),
        lbl("ext:Cartridge","cartridge"),
        lbl("ext:Yield","yield"),
        lbl("ext:usesCartridge","cartridge"),
        lbl("ext:hasYield","yield"),
        lbl("ext:supportsDuplex","duplex unit"),
    ]
    # accessories & cartridges
    for a in ["DuplexUnit","Tray500","WiFiModule"]:
        out += [f"ext:acc_{a} a ext:Accessory .", lbl(f"ext:acc_{a}", a)]
    for i in range(n):
        pr  = f"ext:lx_pr_{i:03d}"
        cart= f"ext:cart_{i:03d}"
        y   = f"ext:yield_{i:03d}"
        pages = [1500, 2500, 3500, 6000, 9000][i % 5]
        acc = f"ext:acc_{['DuplexUnit','Tray500','WiFiModule'][i % 3]}"
        out += [
            f"{pr} a ext:LaserPrinter .", lbl(pr, f"Lex-{i:03d}"),
            f"{cart} a ext:Cartridge .",  lbl(cart, f"E{i:03d}A11E"),
            f"{y} a ext:Yield .",         lbl(y, f"{pages} pages"),
            f"{pr} ext:usesCartridge {cart} .",
            f"{cart} ext:hasYield {y} .",
            f"{pr} ext:supportsDuplex {acc} .",
        ]
    return "\n".join(out) + "\n"

def viessmann_block(n: int) -> str:
    out = []
    out += [
        "ext:Appliance a owl:Class .",
        "ext:Boiler a owl:Class ; rdfs:subClassOf ext:Appliance .",
        "ext:HeatPump a owl:Class ; rdfs:subClassOf ext:Appliance .",
        "ext:PowerRating a owl:Class .",
        "ext:FuelType a owl:Class .",
        "ext:Accessory a owl:Class .",
        "ext:hasOutput a owl:ObjectProperty ; rdfs:domain ext:Appliance ; rdfs:range ext:PowerRating .",
        "ext:supportsFuel a owl:ObjectProperty ; rdfs:domain ext:Appliance ; rdfs:range ext:FuelType .",
        "ext:compatibleWith a owl:ObjectProperty ; rdfs:domain ext:Appliance ; rdfs:range ext:Accessory .",
        lbl("ext:Appliance","appliance"),
        lbl("ext:Boiler","boiler"),
        lbl("ext:HeatPump","heat pump"),
        lbl("ext:PowerRating","power output"),
        lbl("ext:FuelType","fuel type"),
        lbl("ext:Accessory","accessory"),
        lbl("ext:hasOutput","output"),
        lbl("ext:supportsFuel","fuel"),
        lbl("ext:compatibleWith","compatible accessory"),
    ]
    for ft in ["Gas","Oil","Propane","Electric"]:
        out += [f"ext:fuel_{ft} a ext:FuelType .", lbl(f"ext:fuel_{ft}", ft)]
    for a in ["Vitotrol","HydroKit","BufferTank","OutdoorSensor"]:
        out += [f"ext:acc_{a} a ext:Accessory .", lbl(f"ext:acc_{a}", a)]
    kw_vals = [12, 18, 24, 30, 35]
    for i in range(n):
        ap = f"ext:v_ap_{i:03d}"
        cls = "ext:Boiler" if i % 2 == 0 else "ext:HeatPump"
        pr = f"ext:pr_{i:03d}"
        kW = kw_vals[i % len(kw_vals)]
        fuel = f"ext:fuel_{['Gas','Oil','Propane','Electric'][i % 4]}"
        acc  = f"ext:acc_{['Vitotrol','HydroKit','BufferTank','OutdoorSensor'][i % 4]}"
        out += [
            f"{ap} a {cls} .", lbl(ap, f"Vies-{i:03d}"),
            f"{pr} a ext:PowerRating .", lbl(pr, f"{kW} kW"),
            f"{ap} ext:hasOutput {pr} .",
            f"{ap} ext:supportsFuel {fuel} .",
            f"{ap} ext:compatibleWith {acc} .",
        ]
    return "\n".join(out) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["battery","lexmark","viessmann"])
    ap.add_argument("--n", type=int, default=60, help="How many products/entities to add")
    args = ap.parse_args()

    now = datetime.utcnow().strftime("%Y%m%d")
    out = [HDR, f"# KG booster generated {now}\n"]

    if args.domain == "battery":
        out.append(battery_block(args.n))
        out_path = ONT / "battery_augment.ttl"  # will append into dpp_ontology.ttl
    elif args.domain == "lexmark":
        out.append(lexmark_block(args.n))
        out_path = ONT / "lexmark_augment.ttl"
    else:
        out.append(viessmann_block(args.n))
        out_path = ONT / "viessmann_augment.ttl"

    out_path.write_text("".join(out), encoding="utf-8")
    print(f"Wrote {out_path}  (add it to your main TTL as shown in the next step)")

if __name__ == "__main__":
    main()
