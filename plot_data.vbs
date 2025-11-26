Set shell = CreateObject("WScript.Shell")

choice = InputBox("Scegli cosa plottare:" & vbCrLf & _
                  "1 = Vapor" & vbCrLf & _
                  "2 = Liquid" & vbCrLf & _
                  "3 = Wall", "Plot selector")

If choice = "1" Then
    shell.Run "cmd /c python plot_data.py vapor"
ElseIf choice = "2" Then
    shell.Run "cmd /c python plot_data.py liquid"
ElseIf choice = "3" Then
    shell.Run "cmd /c python plot_data.py wall"
Else
    MsgBox "Scelta non valida."
End If
