(defglobal
?*numero* = 7)
(defrule juego
=>
(assert (lee)))

(defrule lee
?h <- (lee)
=>
(retract ?h)
(printout t "Escribe un numero: ")
(bind ?n (read))
(if (not (numberp ?n))
then
(printout t "Eso no es un numero." crlf)
(assert (lee))
else
(assert (numero ?n))))

(defrule bajo
?h <- (numero ?n&:(< ?n ?*numero*))
=>
(retract ?h)
(printout t ?n " es bajo" crlf)
(assert (lee)))

(defrule alto
?h <- (numero ?n&:(> ?n ?*numero*))
=>
(retract ?h)
(printout t ?n " es alto" crlf)
(assert (lee)))

(defrule exacto
?h <- (numero ?n&:(= ?n ?*numero*))
=>
(retract ?h)
(printout t ?n " es correcto" crlf))