# Zeitmessungen für die Auswertung symbolisch berechneter Matrizen

## Einleitung

Im Rahmen dieser Arbeit soll die Anwendbarkeit symbolisch generierter Lösungen zur Anwendung in iterativen Verfahren am Beispiel des eigens implementierten Newton-Verfahrens überprüft werden werden. Dazu wird die, für das Newton-Verfahren notwendige Jacobimatrix mithilfe der Library `Sympy` auf verschiedenen Wegen symbolisch bestimmt und zur Lösung des Newton-Verfahrens für verschieden große Matrizen angewendet. Um einen realistischen Vergleichsfall zu erhalten wird dabei die gesamte notwendige Rechenzeit zur Lösung des nichtlinearen Gleichungssystems, in welche die Zeit zur symbolischen Berechnung der Jacobi-Matrix einfließt, bestimmt und verglichen. Anhand der erzielten Lösungen soll die Laufzeit im Bezug auf die Anzahl an Gleichungen/Variablen des nichtlinearen Gleichungssystems in einem funktionellen Zusammenhang eingeordnet und im Bezug auf eine realistische Anwendbarkeit symbolischer Berechnungen im Sinne der Rechenzeiten untersucht werden.

## Vorüberlegungen

### Vorstellung des Moduls `Sympy`

Bei `Sympy` handelt es sich um eine, vollständig in Python geschriebene, Python-Library, die zahlreiche Funktionen eines *Computer Algebra Systems* mithilfe symbolischer Rechnungen umsetzt und auf ein eigenes Framework zur Speicherung von `Sympy`-Matrizen, `Sympy`-Vektoren sowie `Sympy`-Parametern (gen. `Sympy`-Symbols) bietet. Im Rahmen dieser Arbeit wird primär die Methode `jacobian()`betrachtet, die aus aus einer `Sympy`-Matrix mithilfe von übergebenen `Sympy`-Parametern, die ebenfalls als eine solche Matrix vorliegen, symbolisch eine Jacobi-Matrix erzeugt, die anschließend durch das Einsetzen von Zahlenwerten an Stelle der Variablen ausgewertet werden kann.

### Datentypen

Als numerisch günstigste Wahl des Datentypen würde sich vermutlich die Verwendung von `Numpy`-Arrays anbieten. Allerdings haben diese im Bezug auf die Verwendung von `Sympy` einen entscheidenden Nachteil. Der Datentyp eines `Numpy`-Arrays wird global für das gesamte Array festgelegt und i.d.R. auf `np.float64` eingestellt. Um die Kompatibilität mit `Sympy`-Symbols aufrechtzuerhalten  wäre es notwendig den Datentypen der betroffenen `Numpy`-Arrays auf `object` zu stellen, was potenziell zu einer Verschlechterung der Rechenzeiten beitragen könnte. 

Um dieses Verhalten zu bewerten, wurde im Rahmen der Vorüberlegungen eine eine einfache Zeitmessung durchgeführt, in welcher $n$ zufällige Zahlenwerte aufsummiert werden, die jeweils als float-Werte in einer Liste gespeichert sind oder in einem `Numpy`-Array mit dem Datentypen `float` oder `object` stehen. Das Ergebnis dieser Zeitmessung wird in dernachfolgenden Abbildung dargestellt. Wie zu erwarten, werden die kürzesten Rechenzeiten mit `Numpy`-Array mit dem Datentyp `float` erzielt. Überraschend ist jedoch, dass die Rechenzeiten für Listen mit Einträgen des Datentyps `float` generell geringer ausfallen, als `Numpy`-Arrays mit dem Datentyp `object`. Um die Kompatibilität mit `Sympy` zu gewährleisten und hohe Rechenzeiten zu vermieden, werden daher im weiteren Verlauf der Zeitmessungen Listen verwendet.

Die Ergebnisse dieser Zeitmessung weisen jedoch auch darauf hin, dass für eine Referenzmessung mit dem Newton-Verfahren mit vorgegebener Jacobi-Matrix eine kürzere Rechenzeit bei der Verwendung von `Numpy`-Arrays an Stelle von Listen erzielt werden könnte. An dieser Stelle soll das allerdings vernachlässigt werden, da für die isolierte Betrachtung von `Sympy` ein Vergleich unter identischen Rahmenbedingungen durchgeführt werden soll. Zusätzlich zu diesem Fall werden auch noch Messwerte für den Einsatz einer manuellen Jacobi-Matrix unter Verwendung von Numpy, aufgenommen, um einen möglichst realistischen Vergleichsfall zu schaffen.<img src="C:\Users\sasch\OneDrive\_Documents\_Dokumente\_Studium\21_WiSe\SHK\symbolic-jacobian-times-measurement\graphics\Vergleich_numpy_Datentypen.svg" alt="Vergleich_numpy_Datentypen" style="zoom:150%;" />

### Zeitmessung

Um einen möglichst realistischen Vergleich der Laufzeiten verschiedener Anwendungen der iterativen Lösung eines nichtlinearen Gleichungssystems unter Verwendung einer symbolisch generierten Jacobi-Matrix zu ermöglichen, bezieht sich die Laufzeitmessung auf den gesamten Vorgang der iterativen Berechnung, jedoch nicht auf die Initialisierung dieser. Folglich werden die symbolische Erzeugung der Jacobi-Matrix, die Auswertung dieser in jeder Iteration sowie die Vorgänge in den einzelnen Iterationen des Newton-Verfahrens in der Zeitmessung berücksichtigt. Aus diesem Grund wird auf eine getrennte Laufzeitmessung der einzelnen Vorgänge in der Auswertung verzichtet.

Im Rahmen der Vorüberlegungen wurden derartige gezielte Zeitmessungen einzelner Rechenschritte jedoch bedarfsabhängig ausgeführt, mit dem Ergebnis, dass die benötigte Zeit zur symbolischen Berechnung der Jacobi-Matrix im Hinblick auf die gesamte Laufzeit nicht so schwer ins Gewicht fällt, wie die Auswertung der Jacobi-Matrix, da die Erzeugung, anders als die Auswertung, nur ein einziges Mal und nicht in jeder Iteration erneut durchgeführt werden muss. Diese Messungen dienen jedoch nur als Ansatzpunkt weiterer Optimierungen und die Werte der gemessenen Laufzeiten finden im Weiteren keine Berücksichtigung.

## Aufbau des Newton-Verfahrens

### Grundlagen

Zum Einsatz kommt das eigens implementierte Newton-Verfahren, dass bereits im Rahmen der Projektarbeit "Newton-Verfahren" vom 19. Mai 2021  erstellt wurde. Der grundlegende Aufbau des Programms ist in der nachfolgenden Abbildung in der Form eines UML-Aktivitätsdiagramms dargestellt, die ebenfalls in der genannten Projektarbeit zu finden ist. Der vollständige Quellcode dieses Verfahren ist im Anhang als `src_newton.py` einzusehen.<img src="C:\Users\sasch\OneDrive\_Documents\_Dokumente\_Studium\21_SoSe\Numerische Verfahren\Projekt 2\activity diagram.svg" alt="activity diagram" style="zoom:75%;" />

### Verwendung symbolischer Berechnungen

Während in der o.g. Projektarbeit bereits eine rudimentäre Version der symbolischen Berechnung der Jacobi-Matrix implementiert wurde, findet sich in dem hier verwendeten Quellcode `src_newton.py` eine optimierte Version, die zwischen verschiedenen Verfahren zur Bereitstellung der Jacobi-Matrix für die Auswertung unterscheidet und in der Initialisierung der Berechnung einen Parameter `sympy_method` zulässt, über den das entsprechende Verfahren ausgewählt werden kann. Im Folgenden werden die verschiedenen Verfahren im Detail betrachtet.

#### Auswertung mit `subs()` und `eval()` 

Bei diesem Verfahren handelt es sich um den default zur Auswertung der symbolisch erzeugten Jacobi-Matrix. Die Jacobi-Matrix enthält `Sympy`-Symbols und wird in Form einer `Sympy`-Matrix bereitgestellt. Im Rahmen der Auswertung wird ein Dictionary aus den `Sympy`-Symbols und den einzusetzenden Zahlenwerten erstellt. Mithilfe der Methode `subs()` können nun die `Sympy`-Symbols durch die Zahlenwerte aus dem Dictionary innerhalb der `Sympy`-Matrix ersetzt werden. Die Funktion `evalf()` wertet die Funktion anhand der enthaltenen Zahlenwerte aus und stellt die Lösung bereit.

Gemäß Dokumentation ist dieses Verfahren das langsamste und nur im Rahmen von Prototypen aber nicht zur endgültigen Anwendung empfehlenswert.

#### Auswertung mit `lambdify` und der `math`-Library

Die Funktion `lambdify()` erlaubt es die Jacobi-Matrix, die als `Sympy`-Matrix vorliegt unter Einbindung der `Sympy`-Symbols in eine lambda-Funktion umzuwandeln, die entsprechend per default mit der `math`-Library und nicht mehr mit dem eigenen Framework von `Sympy` ausgewertet werden können. Dieses Vorgehen verspricht Rechenzeiten, die etwa um den Faktor 100 schneller sein sollten als die Auswertung mit `subs` und `eval()`.

#### Auswertung mit `lambdify` und der `numpy`-Library

Auch hier wird wieder die Funktion `lambdify()` eingesetzt, allerdings unter Einbindung von `numpy`. Dementsprechend sind noch geringere Rechenzeiten  zu erwarten, als bei der Verwendung der `math`-Library.

#### Ausblick auf weitere Anwendungen von `lambdify`

Die `math`-Library und `numpy` sind nicht die einzigen numerischen Backends, die sich mit der `lambdify()`-Funktion anwenden lassen. Ebenfalls eine Option wäre `cupy` zur Verwendung von GPU-basierten Berechnungen. Als Alternativen zu `lambdify()` könnten auch `ufuncify`,  `autowrap` oder `binary_function` verwendet werden, um automatisch generierten Code zu speichern und zu kompilieren und die Ergebnisse zu importieren. `Sympy` lässt sich ebenfalls mithilfe von `Aesara` zur Auswertung der Jacobi-Matrix verwenden. Diese Möglichkeiten werden im Rahmen dieses Berichts allerdings keine weitere Verwendung finden, um den Umfang der Untersuchungen einzugrenzen.

### Implementierung symbolischer Berechnungen

Die Implementierung der verschiedenen Vorgehen zur symbolischen Berechnung der Jacobi-Matrix erfolgen im Wesentlichen in der Methode `create_jacobian` und der Methode `filled_jacobian` in `src_newton.py`. Dabei ist die erste für die Erzeugung und die zweite für die Auswertung zuständig.

#### Erzeugung der Jacobi-Matrix

```python
    def create_jacobian(self):
        f = self.equation_system(self.jacobian_symbols, *self.args)
        sympy_matrix = Matrix(f)
        sympy_params = Matrix([self.jacobian_symbols])
        empty_jacobian = sympy_matrix.jacobian(sympy_params)
        if self.sympy_method == 'sympy':
            return empty_jacobian
        elif self.sympy_method == 'math':
            empty_jacobian_lambdifed = lambdify(self.jacobian_symbols, empty_jacobian)
            return empty_jacobian_lambdifed
        elif self.sympy_method == 'numpy':
            empty_jacobian_lambdifed = lambdify(self.jacobian_symbols, empty_jacobian, self.sympy_method)
            return empty_jacobian_lambdifed
```

Bei den `jacobian_symbols` handelt es sich um `Sympy`-Symbols, also Variablen, die in diesem Fall das Format `x_i` haben. Im Rahmen der Erzeugung der Jacobi-Matrix werden erfolgt eine Auswertung des in `equation_sytem` bereitgestellten Gleichungssystems mithilfe der `jacobian-symbols` und optionaler weiterer Parameter (die nicht als `Sympy`-Symbols vorkommen) und anschließend eine Umwandlung in eine `Sympy`-Matrix und die Speicherung als `sympy_matrix`. Auch die `jacobian_symbols` werde in eine `Sympy`-Matrix namens `sympy_params` umgewandelt. Anschließend erfolgt die eigentliche symbolische Erzeugung der Jacobi-Matrix, indem die `sympy_matrix` mithilfe der Methode `jacobian` an den übergebenen `sympy_params` ausgewertet wird.

 Soll die Auswertung mit `subs()` und `evalf()` erfolgen, so wird als Parameter `sympy_method` der String `sympy` übergeben, der ebenfalls per default eingestellt ist. In diesem Fall die eben beschriebene erzeugte leere Jacobi-Matrix übergeben. Für den Fall, dass `lambdify` mit der `math`-Library oder der `numpy`-Library verwendet werden soll, wird als `sympy_method` der String `math` oder `numpy` übergeben. In diesem Fall wird auf die leere Jacobi-Matrix noch die Funktion `lambdify` angewendet, indem die `jacobian_symbols`, die leere Jacobi-Matrix und die zu verwendende Library übergeben werden. Die `math`-Library ist dabei default, sodass in diesem Fall keine Library angegeben werden muss.

#### Auswertung der Jacobi-Matrix

```python
    def filled_jacobian(self, params):
        if type(self.jacobian_symbols) == str:  # in case of manual jacobian
            return np.array(self.jacobian(params, *self.args), dtype=complex)
        if self.sympy_method == 'sympy':
            assigned_params = dict(zip(self.jacobian_symbols, params))
            filled_jacobian = self.jacobian.subs(assigned_params).evalf()
            return np.array(filled_jacobian.tolist(), dtype=complex)
        else:
            return self.jacobian(*params)
```

Bei der Auswertung der Jacobi-Matrix muss zunächst der Fall einer manuell erzeugten Jacobi-Matrix abgefangen werden. In diesem Fall erfolgt die Auswertung klassisch. Andernfalls muss zwischen der mit `sympy` und der mit `lambdify` erzeugten Jacobi-Matrix unterschieden werden.

Im ersten Fall werden die `jacobian_symbols` mit den `params`, also den eigentlichen Zahlenwerten mittels `zip` verbunden und in einem Dictionary gespeichert. Anschließend können die `Sympy`-Symbols der leeren Jacobi-Matrix mithilfe der Methode `subs()` durch die Zahlenwerte ausgetauscht werden, indem das Dictionary übergeben wird. Die Methode `evalf()` wird verwendet, um die Jacobi-Matrix mit den nun eingetragenen Zahlenwerten auszuwerten. Die Lösung liegt jetzt noch als `Sympy`-Vektor vor und kann erst in eine Liste und dann in ein `Numpy`-Array umgewandelt werden.

Wurde die Jacobi-Matrix mit `lambdify` erzeugt, so ist die Auswertung weitaus unkomplizierter. Die Zahlenwerte `params` können einfach mit dem einzelnen Sternchen zum Entpacken der Liste an die Funktion, die mit `lambdify` aus der leeren Jacobi-Matrix gebildet wurde, übergeben werden. Ein weitere Umwandlung des Ergebnisses ist hinfällig, da die Ausgabe bereits in Form eines `Numpy`-Arrays erfolgt.

## Umsetzung der Zeitmessung

### Gleichungssystem zur Zeitmessung

Das verwendete Gleichungssystem wird dem zur Zeitmessung bereitgestellten Code `Spheres.py` (siehe Anhang) entnommen. Dabei erfolgt die Erstellung des Gleichungssystems über die Funktion `funcIntersectingSpheres`, die nachfolgend abgebildet ist.

```python
    def funcIntersectingSpheres(x, *data):
        n = x.shape[0]
        f, J, r = data

        for k in range(n):
            f[k] = -r[k] ** 2
            for j in range(n):
                if j == k:
                    f[k] += (x[j] - r[k]) ** 2
                else:
                    f[k] += x[j] ** 2
        return f
```

Für die Verwendung einer manuell bestimmten Jacobi-Matrix wird die ebenfalls bereitgestellte Funktion `JIntersectingSpheres` verwendet:

```python
    def JIntersectingSpheres(x, *data):
        n = x.shape[0]
        f, J, r = data

        for k in range(n):
            J[k] = 2 * x
            J[k][k] += 2 * (-r[k])
        return J
```

Bei der Variable `f` handelt es sich um einen Nullvektor der Länge $n$. Die Variable `J` setzt sich aus einer Nullmatrix der Dimension $n \times n$ zusammen. Die Variable `r` wird mit einem Vektor aus Einsen mit der Länge $n$ erzeugt. Aufgrund der Abhängigkeit von $n$, müssen diese Variablen für jedes $n$ neu erzeugt werden. 

### Vorgehen zur Zeitmessung

Zur Ausführung der Zeitmessung wird die Rechenzeit der Approximation der Lösung für die oben verschiedenen Methoden für verschiedene $n$ mehrfach ausgeführt. Dazu wird das Intervall von $n$ inklusive der Schrittweite sowie die Anzahl an Wiederholungen zur Mittelwertsbildung per Nutzerabfrage bestimmt. Aus diesen Werten eine Liste mit den entsprechenden $n$ erzeugt, die sich gemäß der Anzahl an Wiederholungen zur Mittelwertsbildung aufsteigend wiederholt (siehe `np.tile()`). Diese Art der Wiederholung wird dabei gegenüber einer direkt aufeinanderfolgenden Wiederholung der einzelnen Werte (siehe `np.repeat()`) bevorzugt, um den Einfluss möglicherweise noch im Cache vorhandener Werte zu reduzieren.

Innerhalb einer Schleife, die durch die Liste der $n$ iteriert wird das verfahren für alle vier Methoden ausgeführt. Dabei wird das Verfahren immer erst initialisiert, bevor die Zeitmessung gestartet wird. Die Zeitmessung umfasst lediglich die Methode der Approximation. Für jedes Verfahren wird die gemessene Zeit einer Liste hinzugefügt.

Die gemessenen Rechenzeiten werden insofern aufbereitet, als das aus den Wiederholungen für jedes Verfahren und für jedes $n$ der Mittelwert gebildet und der Standardfehler berechnet wird. Es resultiert für jedes Verfahren also jeweils eine Liste mit den Mittelwerten und eine Liste mit den Standardfehlern, die in zweidimensionalen `Numpy`-Arrays für alle Verfahren zusammen gespeichert werden.

## Ergebnisse der Zeitmessung

### Grafische Auswertung der Rechenzeiten

Die Ergebnisse der Zeitmessung können grafisch dargestellt werden, um die Abhängigkeit zwischen der Größe $n$ des Gleichungssystems und der Rechenzeit für jedes Verfahren zu untersuchen. Um die Rechenzeiten in einem große Bereich von $n$ sinnvoll abzubilden erfolgt die grafische Darstellung in einem logarithmisch-linearen Plot. Eine ebenfalls logarithmische Darstellung der $x$-Achse (Größe $n$ des Gleichungssystems) wäre an dieser Stelle zwar durchaus sinnvoll, aber aufgrund der Rechenzeiten für große $n$ bei manchen der untersuchten Verfahren nicht praktikabel. Die so erzeugte Grafik ist nachfolgend abgebildet. Es wurde der Bereich von $n=5$ bis $n = 100$ mit einer Schrittweite von 5 und  jeweils 5 Wiederholungen zur Mittelwertbildung verwendet.

<img src="C:\Users\sasch\OneDrive\_Documents\_Dokumente\_Studium\21_WiSe\SHK\symbolic-jacobian-times-measurement\graphics\vollständiger_Vergleich_mit_Mittelwerten_und_Fehler_neu.svg" alt="vollständiger_Vergleich_mit_Mittelwerten_und_Fehler_neu" style="zoom:200%;" />

An der Abbildung ist abzulesen, dass sich die Rechenzeiten für die reine Verwendung von `Sympy` mit den Methoden `subs()` und `evalf()` ausnahmelos in einem weitaus höheren Bereich, als bei allen anderen getesteten Verfahren befinden. Die Werte liegen in einer Größenordnung von $10^{-1}$ s bis $10^3$ s. Bei der Verwendung der `lambdify`-Funktion mit der `math`-Library bewegen sich die Rechenzeiten in einem Bereich von $10^{-3}$ s bis $10^{-1}$ s und liegen damit etwa gleichauf mit den Rechenzeiten für die Auswertung einer manuell bereitgestellten Jacobi-Matrix in Form einer Liste oder eines `numpy`-Arrays. Die niedrigsten Rechenzeiten ergeben sich bei der Verwendung der `lambdify`-Funktion mit der `numpy`-Library. Hier liegen die gemessenen Zeiten in einem Bereich von $10^{-4}$ s bis $10^{-2}$ s. Der zeitliche Vorteil bei der Verwendung der `lambdify`-Funktion mit der `numpy`-Library gegenüber der manuell bereitgestellten Jacobi-Matrix mit `numpy`-Arrays ist darauf zurückzuführen, dass in der manuellen Bereitstellung Schleifen zum Einsatz kommen. Durch eine numerisch günstigere Form der Bereitstellung der manuellen Jacobi-Matrix könnte entsprechend an dieser Stelle Rechenzeit eingespart werden. Die verwendeten Methoden ordnen in Bezug auf ihre Rechenzeiten untereinander gemäß der in der Dokumentation beschriebenen zu erwartenden Reihenfolge ein.

Der Standardfehler wird durch die Errorbars dargestellt, die für jeden Messpunkt eingezeichnet sind. Anhand der geringen vertikalen Ausdehnung der Errorbars ist festzustellen, dass der Standardfehler der meisten Messpunkte im Bezug auf die allgemeine Größenordnung der betrachteten Werte als vernachlässigbar erachtet werden kann. Zu beachten sei an dieser Stelle, dass der Standardfehler auch der Verzerrung durch die logarithmische Skalierung der $y$-Achse unterliegt. 

Die Methode mit der reinen Verwendung von `Sympy` disqualifiziert sich aufgrund der Größenordnung ihrer Rechenzeiten für die meisten Anwendungen numerischer Berechnungen, beispielsweise die Verwendung im Newton-Verfahren, insbesondere im Vergleich zu den Methoden, die auf `lambdify` zurückgreifen. Besagte Methoden sehen für solcherlei Anwendungen weitaus vielversprechender aus, vor allem, da sie auf einer Höhe mit der Methode mit manueller Jacobi-Matrix liegen. Für eine abschließende Bewertung fehlt hier allerdings eine Prüfung des realitätsnäheren Anwendungsfalls einer Auswertung der manuell bereitgestellten Jacobi-Matrix über `Numpy`-Arrays in einem numerisch günstigeren Format als mit Schleifen.

 ### Anwendung verschiedener Funktionsmodelle auf die gemessenen Laufzeiten

Mit dem Ziel für alle Methoden einen funktionellen Zusammenhang zwischen der Rechenzeit und der Größe $n$ des Gleichungssystems zu finden, wurde eine Regression mithilfe eines quadratischen und eines kubischen Funktionsmodells durchgeführt. Polynomfunktionen höherer Ordnung wurden nicht getestet, da ihr Vorkommen im Sinne der Laufzeitoptimierung bei mathematikorientierten Softwarepaketen als weitestgehend unbedeutend erachtet wird. Die Tatsache, dass in der Laufzeitmessung nicht nur die Auswertung der Jacobi-Matrix, sondern die gesamte Approximation erfasst wird, bedeutet, dass sich die Anwendung des Funktionsmodells nicht ausschließlich auf die Verwendung von `Sympy` bezieht und dementsprechend Abweichungen herbeiführen kann. Dieser Zusammenhang ist jedoch im Sinne einer realistischen Bewertung der Verwendung von `Sympy` im Rahmen echter numerischer Software sogar wünschenswert. Eine Anwendung eines quadratischen Funktionsmodells auf alle Methoden ist in der nachfolgenden Abbildung dargestellt.

<img src="C:\Users\sasch\OneDrive\_Documents\_Dokumente\_Studium\21_WiSe\SHK\symbolic-jacobian-times-measurement\graphics\2022-03-31T21-15-55_quadratic_fit.svg" alt="2022-03-31T21-15-55_quadratic_fit" style="zoom:150%;" />

Es zeigt sich, dass ein quadratisches Funktionsmodell zur Darstellung des betrachteten Zusammenhangs zumindest für die Methode der reinen Anwendung von `Sympy` gänzlich ungeeignet ist. Aufgrund der besonders hohen Rechenzeiten erscheint das Vorkommen eines Zusammenhangs über eine Polynomfunktion höheren Grades hier durchaus nicht unwahrscheinlich. Allerdings liegt auch bei den anderen Methoden eine nicht zu vernachlässigende Diskrepanz zwischen der Regression durch eine quadratische Funktion und dem tatsächlichen funktionellen Zusammenhang vor. Diese zeigt sich primär für kleinere $n$. Auch an dieser Stelle muss auf die Verzerrungen durch die logarithmische Darstellung hingewiesen werden, aufgrund derer Abweichungen womöglich schwieriger zu erkennen sind.

In der nachfolgenden Abbildung wird die Regression mittels eines kubischen Funktionsmodells durchgeführt. Es ergibt sich für alle Methoden eine weitaus präzisere Approximation des Funktionsverlaufs, als es mit der quadratischen Funktion möglich war. Für die Methode mit Verwendung von reinem `Sympy` zeigen sich, zumindest für kleine $n$, immer noch deutliche Abweichungen des tatsächlichen Funktionsverlaufes, die sich mit dem Vorkommen eines Tiefpunkts bei $n = 10$ beschreiben lassen, der so in den tatsächlichen Messwerten gänzlich nicht vorhanden ist. Von einem kubischen Zusammenhang zwischen der Rechenzeit und $n$ kann hier offenbar nicht ausgegangen werden.

Für die Verwendung von `Sympy` und `lambdify` zeigt sich mit dem quadratischen Funktionsmodell eine durchaus akkurate Abbildung der tatsächlichen Funktionswerte. Im Vergleich zum quadratischen Funktionsmodell wird nun auch der Bereich kleiner Werte für $n$ präziser abgebildet. Das Vorliegen eines kubischen Zusammenhangs scheint hier durchaus naheliegend.

Für die Methoden mit der manuell bereitgestellten Jacobi-Matrix kommt es immer noch zu kleinen Abweichungen, insbesondere für kleine $n$. Diese Abweichungen deuten einen nicht-kubischen funktionellen Zusammenhang hin. Ebenso zeigt sich für die Verwendung von `Sympy` und `math` eine solche Abweichung für kleine $n$.

<img src="C:\Users\sasch\OneDrive\_Documents\_Dokumente\_Studium\21_WiSe\SHK\symbolic-jacobian-times-measurement\graphics\2022-03-31T21-15-56_cubic_fit.svg" alt="2022-03-31T21-15-56_cubic_fit" style="zoom:150%;" />



## Fazit

Im Rahmen dieser Untersuchungen sollte festgestellt werden, inwiefern die Implementation symbolischer Lösungsverfahren über die `Sympy`-Library in Lösern für numerische realistisch ist. Dazu wurden die Laufzeiten für eine Approximation mit dem Newton-Verfahren, bei dem die Jacobi-Matrix über verschiedene Methoden symbolisch errechnet wird, gemessen und mit den Laufzeiten für manuell über Listen oder `numpy`-Arrays bereitgestellten Jacobi-Matrizen verglichen. Es zeigt sich, dass durch Einsatz von `Sympy` zur Bestimmung der Jacobi-Matrix und der Verwendung von `lambdify` und der `Numpy`-Library zur Auswertung die gemessenen Laufzeiten einer manuell über Listen bereitgestellten Jacobi-Matrix sogar noch unterschritten werden können. Die erzielte Zeit zur numerischen Lösung eines Gleichungssystems mit 100 Einträgen durch das Newton-Verfahren liegt gemäß dieser Messungen bei ca. $0,02$ s und damit in einem durchaus vertretbaren zeitlichen Rahmen zur Anwendung für numerische Löser. Es ist anzunehmen, dass sich diese Zeit in Abhängigkeit von $n$ kubisch vergrößern wird, wobei damit nicht die Zeit zur Auswertung der Jacobi-Matrix, sondern die gesamte Zeit für die Approximation mit dem Newton-Verfahren betrachtet wurde. Um jedoch ein abschließendes Fazit zur Verwendung von `Sympy` formulieren zu können fehlt es an einem Vergleich mit einem optimierten und realitätsnahen Anwendungsfall mit manuell bereitgestellter Jacobi-Matrix.

## Perspektiven

Zur Fortsetzung der hier aufgeführten Untersuchungen sollten folgende Verbesserungen der Methodik vorgenommen werden:

- effizientere Bereitstellung der manuell errechneten Jacobi-Matrix, um einen realistischen Vergleichsfall zu schaffen
- Betrachtung der Standardabweichung anstelle des Standardfehlers zur Bewertung der Konsistenz bei der Einzelrechnungen bei der Mittelwertsbildung
- Betrachtung des Standardfehlers und der Funktionsmodelle ohne die Verzerrung einer logarithmischen Darstellung (Approximation von Funktionsmodellen kann auf doppelt logarithmierte Messwerte angewendet werden. So lässt sich eine mögliche Polynomfunktion anhand eines linearen Zusammenhangs nachweisen und untersuchen.)

Weitere Verbesserungen und zu erreichende Ziele können folgende sein:

- Verbesserung des Programms um eine Speicherung der rohen Messdaten vorzunehmen
- Implementierung und Zeitmessungen weiterer numerischer Backends für `Sympy`

## Quellen

- Dokumentation von `Sympy`: https://docs.sympy.org/latest/index.html

- Allgemeine Einführung in die numerischen Hintergründe und Submodule von `Sympy`: https://docs.sympy.org/latest/modules/codegen.html#id1

- Einführung in die `Lambdify`-Funktion (ebenfalls aus der Dokumentation von `Sympy`): https://docs.sympy.org/latest/modules/utilities/lambdify.html
- Empfehlungen zur Laufzeitverbesserung von `Sympy`; numerische Hintergründe der Verfahren; grober Vergleich der Laufzeiten der verschiedenen Verfahren: https://docs.sympy.org/latest/modules/numeric-computation.html
