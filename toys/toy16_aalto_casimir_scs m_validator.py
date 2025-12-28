"""
TOY 16 — AALTO CASIMIR → SCSM VALIDATOR (Full HDF5 explorer)
"""
import h5py
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass

@dataclass
class Toy16Params:
    target_force_pN: float = 52.0
    d_ref_m: float = 1.0e-3
    area_um2: float = 1.0

def plateau_F(d, F0, d0, n):
    return F0 / (1.0 + (d / d0) ** n)

def standard_casimir(d):
    hc = 1.97e-25
    return - (hc * np.pi**2 / 240) / d**4 * 1e12 * Toy16Params.area_um2

def explore_h5_structure(h):
    """Explore full HDF5 structure"""
    def print_structure(name, obj):
        print(f"  {name}: {type(obj)} shape={getattr(obj, 'shape', 'N/A')}")
    
    h.visititems(print_structure)

def main():
    print("=== TOY16: Aalto Casimir → SCSM Validator ===")
    
    files = ['Casimir_pressure_dist_v3.h5', 'Casimir_pressure_dist_v4.h5']
    
    for f in files:
        if os.path.exists(f):
            print(f"\nLoading {f}...")
            with h5py.File(f, 'r') as h:
                print("Full structure:")
                explore_h5_structure(h)
                
                # Handle block0 compound format
                if 'data' in h:
                    data = h['data']
                    block0_items = data['block0_items'][:]
                    block0_values = data['block0_values'][:]
                    
                    print(f"block0_items: {block0_items}")
                    print(f"block0_values shape: {block0_values.shape}")
                    
                    # Assume standard [distance, pressure] ordering
                    if len(block0_items) >= 2:
                        d_nm = block0_values[:, 0]  # First column = distance
                        P_Pa = block0_values[:, 1]  # Second column = pressure
                    else:
                        print("Unexpected format - using first available data")
                        d_nm = data['axis0'][:]
                        P_Pa = data['axis1'][:]
                else:
                    keys = list(h.keys())
                    d_nm = h[keys[0]][:]
                    P_Pa = h[keys[1]][:]
                
                # Convert units
                d_m = d_nm * 1e-9 if d_nm.max() > 10 else d_nm  # Auto-detect nm vs m
                F_pN = np.abs(P_Pa) * 1e-6 * Toy16Params.area_um2  # Pa → pN
                
                # Save clean CSV
                np.savetxt('aalto_clean.csv', np.column_stack([d_m, F_pN]),
                          header='d_m,F_pN', delimiter=',')
                
                print(f"✅ Extracted {len(d_m)} points:")
                print(f"   d: {d_m[0]*1e3:.2f} - {d_m[-1]*1e3:.2f} mm")
                print(f"   F: {F_pN[0]:.2f} - {F_pN[-1]:.2f} pN")
                break
    else:
        print("❌ No .h5 files found!")
        return
    
    # SCSM Fit
    print("\nFitting SCSM plateau...")
    d_mm = d_m * 1e3
    p0 = [60.0, np.median(d_m), 6.0]
    
    try:
        popt, _ = curve_fit(plateau_F, d_m, F_pN, p0=p0, maxfev=10000)
        F0, d0, n = popt
        F1mm = plateau_F(Toy16Params.d_ref_m, *popt)
        
        print(f"\n🎯 SCSM FIT RESULTS")
        print(f"F0 = {F0:.1f} pN")
        print(f"d0 = {d0*1e3:.1f} mm") 
        print(f"n  = {n:.1f}")
        print(f"F(1mm) = {F1mm:.1f} pN  ← TARGET 52 pN")
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.semilogy(d_mm, F_pN, 'ko', markersize=8, label='Aalto Casimir Data')
        d_fit = np.linspace(d_m.min(), d_m.max(), 200)
        plt.semilogy(d_fit*1e3, plateau_F(d_fit, *popt), 'r-', linewidth=4, 
                    label=f'SCSM Plateau F0={F0:.1f}pN')
        plt.semilogy(d_fit*1e3, np.abs(standard_casimir(d_fit)), 'b--', 
                    label='Standard Casimir')
        plt.axvline(1.0, color='orange', lw=2, ls=':', label='1mm Target')
        plt.axhline(52, color='orange', lw=2, ls=':', alpha=0.7, label='52pN')
        plt.xlabel('Distance (mm)', fontsize=12)
        plt.ylabel('Force (pN)', fontsize=12)
        plt.title('SCSM v11.5.3 vs Aalto Casimir Drums\nRxiverse #19119439', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('toy16_scsm_result.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        status = "✅ SUPPORTED" if abs(F1mm - 52) < 15 else "❌ NO SIGNAL (expected scoping)"
        print(f"\n{status}")
        
    except Exception as e:
        print(f"Fit failed: {e}")
        print("Raw data saved as aalto_clean.csv for manual analysis")

if __name__ == "__main__":
    main()
