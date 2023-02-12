# Ejercicio 1
def suma_n_primeros(n):
    total = 0
    for i in range(n):
        division = ((-1) ** i) / ((2 * i) + 1)
        formula = 4 * division
        total = total + formula
    return total


# Ejercicio 2
def tabla_del_n(numero, top):
    return [numero * i for i in range(top)]


# Ejercicio 3
# Hard to test with whitespace.

# Ejercicio 4
# No termino de entender la diferencia entre 4 y 5 solo viendo el código +
# code con break sucks.


# Ejercicio 5
def es_primo(numero):
    if numero < 2:
        return False
    for i in range(2, numero):
        if numero % i == 0:
            return False

    return True


# Ejercicio 6
def primos_inferiores(numero):
    if numero <= 0:
        print("Ha introducido un número negativo, introduzca uno positivo.")
        return []
    if numero == 1 or numero == 2:
        print(f"No existen números primos inferiores a {numero}")
        return []

    return [i for i in range(2, numero) if es_primo(i)]


# Ejercicio 7
def esperanza_de_vida(personas, esperanza):
    return {nombre: age + esperanza[age] for nombre, age in personas.items()}


def main():
    print("\n********* EJERCICIO 1 *********")
    numero = input("Introduzca entero: ")
    n = int(numero)
    total = suma_n_primeros(n)
    print(f"La suma de los primeros {n} términos de la secuencia es: {total}")

    print("\n********* EJERCICIO 2 *********")
    numero = input("Introduzca entero: ")
    n = int(numero)
    print(f"\nLa tabla del {n} es: {tabla_del_n(numero, 11)}")

    print("\n********* EJERCICIO 3 *********")
    numero = input("Introduzca numero de filas: ")
    n = int(numero)
    for i in range(n + 1):
        for j in range(i):
            print(j + 1, end=" ")
            if j == i - 1:
                print(" ")

    print("\n********* EJERCICIO 4 *********")
    numero = input("Introduzca entero: ")
    numero = int(numero)
    print(f"El número {numero} {'es' if es_primo(numero) else 'no es'} primo")

    print("\n********* EJERCICIO 5 *********")
    print(es_primo(11))
    print(es_primo(9))
    print(es_primo(2))

    print("\n********* EJERCICIO 6 *********")
    numero = input("Introduzca entero: ")
    num = int(numero)
    primos = primos_inferiores(num)
    print(f"Primos inferiores a {num}: {primos}")

    print("\n\n********* EJERCICIO 7 *********")
    PERSONAS = {"pedro": 28, "maría": 21, "marta": 22}
    ESPERANZA = {28: 53.4, 21: 65.6, 22: 64.5}

    edv = esperanza_de_vida(PERSONAS, ESPERANZA)
    print(f"La esperanza de vida de Pedro es de {edv['pedro']} años.")
    print(f"La esperanza de vida de María es de {edv['maría']} años.")
    print(f"La esperanza de vida de Marta es de {edv['marta']} años.")


if __name__ == "__main__":
    main()
