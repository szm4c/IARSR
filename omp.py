import numpy as np
import numbers

def omp(Phi, s, m=None, eps=None):
    '''
    Funkcja jest implementacja algorytmu orthogonal matching pursuit w Pythonie za pomoca biblioteki numpy.
    Przyjmowane argumenty to:
    - Phi - macierz, ktorej kolumnami są wektory ze słownika
    - s - sygnal, ktory chcemy przyblizyc, wymagane jest aby byl wektorem pionowym - s.shape = (d, 1)
    - m - jak rzadki ma byc wygenerowany wektor
    - eps - dokladnosc z jaka chcemy przyblizyc sygnal, skalar
    - stop - ilosc iteracji po ktorych algorytm ma sie zatrzymac
    '''

    # wstepna obsluga argumentow
    try:
        Phi = np.array(Phi, dtype=np.complex128)     # najpierw sprawdzam czy Phi da sie przedstawic w formie np.array
    except:
        raise ValueError('Argument Phi jest niepoprawny. Powinien byc macierza o wymiarze: d x N.')

    try:
        s = np.array(s, dtype=np.complex128)     # najpierw sprawdzam czy Phi da sie przedstawic w formie np.array
    except:
        raise ValueError('Argument s jest niepoprawny. Powinien byc wektorem o wymiarze: d x 1.')

    if not (eps == None):
        try:
            if not isinstance(eps, numbers.Number):     # sprawdzam czy eps jest liczba
                raise ValueError('Argument eps jest niepoprawny. Powinien być dodatnia liczba rzeczywista.')
            elif not (eps > 0):                         # sprawdzam czy eps jest dodatni
                raise ValueError('Argument eps jest niepoprawny. Powinien być dodatnia liczba rzeczywista.')
        except:                                         # wszystko jest w bloku try, except poniewaz porownanie liczby zespolonej z zerem w drugim kroku wyrzuci TypeError    
            raise ValueError('Argument eps jest niepoprawny. Powinien być dodatnia liczba rzeczywista.')

    # sprawdzenie wymiaru Phi i s
    s = s.reshape(s.shape[0], 1)    # upewniam sie, ze s bedzie mial dwa wymiary (w ten sposob moze byc wektorem pionowym)

    if not (Phi.shape[0] == s.shape[0]):    # sprawdzam czy ilosc wierszy w Phi i s jest taka sama
        raise ValueError('Liczba wierszy w macierzy Phi i wektorze s musza byc rowne.')
    elif not (s.shape[1] == 1):             # sprawdzam czy s jest wektorem
        raise ValueError('Argument s jest niepoprawny. Powinien byc wektorem o wymiarze: d x 1.')


    # faza przygotowawcza:
    d = s.shape[0]      # wymiar pionowy wektora s (ile ma rzedow)
    N = Phi.shape[1]    # ilosc atomow w slowniku

    if m == None:
        m = N

    # zerowy krok
    a = np.zeros((d,1), dtype=np.complex64)                 # poczatkowe przyblizenie
    r = s                               # poczatkowe residuum
    Lambda = []                         # zbior indeksow atomow, ktore zostaly wybrane do k-tego kroku wlacznie
    # print(f'Zerowy krok.')
    # print(f'Macierz (slownik) Phi = {Phi}')
    # print(f'Wektor s = {s}')


    for i in range(1, m+1):
        # print(f'Rozpoczynam {i} iteracje.')
        # i-ty krok
        l = np.abs( np.conj(r.T) @ Phi ).argmax()         # wybranie indeksu atomu, który najbardziej koreluje z residuum     
        Lambda.append(l)                         # dodanie do zbioru indeksow tego, ktory zostal wybrany w k-tym kroku
        # print(f'Zbior indeksow Lambda = {Lambda}')

        b, _, _, _ = np.linalg.lstsq(Phi[:, Lambda], s, rcond=None)     # funkcja zwraca jeszcze 'residuals', 'rank', 's', ale ich nie potrzebuje
        # print(f'Wektor b = {b}')
        # policzone wyzej b jest wektorem wspolczynnikow bioracych udzial w reprezentacji a_k
        # z dokumentacji wynika ze b.shape = (i, 1)

        a = Phi[:, Lambda] @ b          # aktualizujemy a_k mnożąc Phi_opt z wektorem wspolczynnikow b, Phi[:, Lambda].shape = (d, i), b.shape = (i, 1)
        # print(f'Wektor a = {a}')
        r = s - a                       # aktualizujemy residuum
        # print(f'Wektor r = {r}')

        # dodatkowy warunek przerwania petli
        if not (eps == None):
            if np.abs( np.conj(r.T) @ r ) <= eps:
                print('Zakończenie pracy ponieważ residuum jest mniejsze od eps.')
                break
    
    # na tym etapie wektor b nie zawsze musi miec wymiar N x 1 (moze byc mniejszy)
    # sprowadzenie wektora b do odpowiedniego wymiaru

    b_full = np.zeros((N,1), dtype=np.complex128)
    b_full[Lambda,:] = b
    
    return b_full     