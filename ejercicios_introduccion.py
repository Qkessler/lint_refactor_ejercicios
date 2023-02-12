#Ejercicio 1
print('\n********* EJERCICIO 1 *********')
numero=input('Introduzca entero: ')
n=int(numero)
total=0
for i in range(n):
    division=((-1)**i)/((2*i)+1)
    formula=4*division
    total=total+formula
    
print(f'La suma de los primeros {n} términos de la secuencia es: {total}')


#Ejercicio 2
print('\n********* EJERCICIO 2 *********')
numero=input('Introduzca entero: ')
n=int(numero)

print(f'\nLa tabla del {n} es: ')
for i in range(11):
    mult=n*i
    print(mult)



#Ejercicio 3
print('\n********* EJERCICIO 3 *********')
numero=input('Introduzca numero de filas: ')
n=int(numero)
for i in range(n+1):
    for j in range(i):
        print(j+1, end=' ')
        if(j==i-1):
            print(' ')


#Ejercicio 4
print('\n********* EJERCICIO 4 *********')
numero=input('Introduzca entero: ')
num=int(numero)
if num<=0:
    print("Ha introducido un número negativo, introduzca uno positivo.")

if num==1:
    print("El número 1 no es un número primo")
if num==2:
    print("El número 2 es un número primo")
else:
    for i in range(2, num):
        if num % i == 0:
            print(f"El número {num} no es primo")
            break
    if i==num-1:        ##Hago esto porque si i=num-1 significa que ha recorrido el bucle anterior sin encontrar ningún resto=0. En caso de no poner el if, mi programa solo detectaría números no primos divisibles por 2.
        print(f"El número {num} es primo")
            
    



#Ejercicio 5
print('\n********* EJERCICIO 5 *********')
def es_primo(numero):
    if numero<2:
        return False
    for i in range(2, numero):
        if numero % i == 0:
            return False
            
        
    return True
        

print(es_primo(11))
print(es_primo(9))
print(es_primo(2))


##Ejercicio 6
print('\n********* EJERCICIO 6 *********')
numero=input('Introduzca entero: ')
num=int(numero)
if num<=0:
    print("Ha introducido un número negativo, introduzca uno positivo.")

if num==1 | num==2:
    print(f"No existen números primos inferiores a {num}")
else:

    print(f"Los números primos inferiores a {num} son: ")
    for i in range(2,num):
        if es_primo(i):
         print(i, end=' ' )
    

#Ejercicio 7
print('\n\n********* EJERCICIO 7 *********')
personas={
    'pedro':28,
    'maría':21,
    'marta':22
}

esperanza_adicional={
    28:53.4,
    21:65.6,
    22:64.5
}

personas_esperanza={
    nombre: edad + esperanza_adicional[edad] for nombre, edad in personas.items()
}
print(f"La esperanza de vida de Pedro es de {personas_esperanza['pedro']} años.")
print(f"La esperanza de vida de María es de {personas_esperanza['maría']} años.")
print(f"La esperanza de vida de Marta es de {personas_esperanza['marta']} años.")