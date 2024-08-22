#!/usr/bin/env python3

import sys

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.optimize import curve_fit
import csv
import scipy.integrate as integrate
import scipy.stats as stats


if len(sys.argv) == 1:
    print("Usage: python UniMuonV2.py  muons.txt ...")
    sys.exit(1)
n_files=len(sys.argv)

energyDistribution,AzimutalDistribution,zenitalDistribution= np.loadtxt(sys.argv[1])
if(n_files>2):
    i=2
    while(i<n_files):
        arreglo1,arreglo2,arreglo3=np.loadtxt(sys.argv[i])
        energyDistribution=np.concatenate((energyDistribution, arreglo1))
        zenitalDistribution=np.concatenate((zenitalDistribution, arreglo3))
        AzimutalDistribution=np.concatenate((AzimutalDistribution, arreglo2))
        i=i+1
        
print("La cantidad de datos importados fueron "+str(len(energyDistribution)))        
def costhetaN(O):
    p1,p2,p3,p4,p5=0.102573, -0.068287, 0.958633,0.0407253,0.817285   
    return (0.14*((((np.cos(O))**2)+(p1)**2+ p2*(np.cos(O))**(p3)+p4*(np.cos(O))**(p5))/(1+(p1)**2+p2+p4))**(1/2))
    
def fluxPaperh(E,O,h):
    return (0.14*(E*(1+(3.64)/(E*(costhetaN(O))**(1.29))))**(-2.7)*((1/(1+(1.1*E*costhetaN(O))/(115)))+(0.054/(1+(1.1*E*costhetaN(O))/(850))))*np.exp(h/(4900+750*E)))


def mostrar_menu(opciones):
    print("Seleccione una opcion:")
    for clave in sorted(opciones):
        print(str(clave) +" : "+ opciones[clave][0])
        


def leer_opcion(opciones):
    a = input('Opcion: ')
    if a.isdigit():
        numero = int(a)
        if 1 <= numero <= 6:
            print("El numero ingresado es valido.")
        else:
            print("El numero debe estar entre 1 y 6.")
    else:
           print("La entrada no es un numero entero valido.")
    return a


def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()


def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()


def menu_principal():
    opciones = {
        '1': ('Distribucion del momentum', accion1),
        '2': ('Distribucion azimutal', accion2),
        '3': ('Distribucion del angulo zenital', accion6),
        '4' : ('Mostar fit del angulo zenital', accion3),
        '5' : ('Guardar los datos en un txt', accion4),
        '6': ('Salir', salir)
    }

    generar_menu(opciones, '6')


def accion1():
    mon=energyDistribution
    plt.figure()
    plt.hist(mon,bins=100)
    plt.yscale("log")
    plt.title("Distribucion de momento",fontsize=16)
    plt.xlabel("Momentum (GeV)",fontsize=14)
    rout=input("Nombre del archivo de salida")
    plt.savefig(rout+".png")
    print('Las figuras se han guardado exitosamente')


def accion2():
    plt.figure()
    azi=AzimutalDistribution
    plt.clf()
    plt.hist(azi,bins=100)
    plt.title("Distribucion de angulo Azimutal",fontsize=16)
    plt.xlabel("Angulo (radianes)",fontsize=14)
    rout=input("Nombre del archivo de salida")
    plt.savefig(rout+".png")
    print('Las figuras se han guardado exitosamente')
    
def accion6():
    plt.figure()
    zen=zenitalDistribution
    plt.clf()
    plt.hist(zen,bins=97)
    plt.title("Distribucion de angulo Zenital",fontsize=16)
    plt.xlabel("Angulo (radianes)",fontsize=14)
    rout=input("Nombre del archivo de salida")
    plt.savefig(rout+".png")

    print('Las figuras se han guardado exitosamente')

def accion4():
    mon=energyDistribution
    azi=AzimutalDistribution
    zen=zenitalDistribution
    data=[mon,azi,zen]
    fmt = '%.3f'

    ruta_archivo = "array.txt"

    np.savetxt("muones.txt", data, fmt=fmt)

    
def accion3():
    zen=zenitalDistribution
    hist, bin_edges = np.histogram(zen, bins=85, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def cosenos(x, A,w,n):
        return A*(np.cos(w*x)**n)

    popt, pcov = curve_fit(cosenos, bin_centers, hist)
    perr = np.sqrt(np.diag(pcov))
    mu_fit, sigma_fit,n_fit = popt
    print("El ajuste a realizar es A*(cos(wx))^{n}")
    print("Parametros del ajuste son")
    print("A="+str(mu_fit)+ " +- "+ str(perr[0]))
    print("W="+str(sigma_fit)+ " +- "+ str(perr[1]))
    print("n="+str(n_fit) + " +- " + str(perr[2]))
    x=zen
    plt.figure()
    plt.clf()

    plt.hist(x, 85, density=True, alpha=0.5, label='Histograma')
    plt.plot(bin_centers, cosenos(bin_centers, mu_fit, sigma_fit,n_fit), 'r-', label='Ajuste')
    plt.xlabel('Valores')
    plt.ylabel('Densidad')
    plt.legend()
    plt.savefig("Ajuste.png")
    
    
    print('Has elegido la opcion 3')
    
def salir():
    print('Saliendo')


if __name__ == '__main__':
    menu_principal()

