package mybots;

import robocode.*;
import robocode.util.Utils;

import java.io.*;
import java.net.Socket;
import java.util.Locale;

public class SmartShooterBot extends AdvancedRobot {
    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;

    public void run() {
        setAdjustRadarForGunTurn(true);
        setAdjustGunForRobotTurn(true);

        connectToPythonServer();

        // Nieskończona rotacja radaru i działa
        while (true) {
            setTurnRadarRight(Double.POSITIVE_INFINITY);
            setTurnGunRight(20);  // powoli obracaj działo niezależnie
            execute();
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        double myX = getX();
        double myY = getY();
        double gunHeading = getGunHeadingRadians();

        double distance = e.getDistance();
        double bearing = e.getBearingRadians();
        double absBearing = getHeadingRadians() + bearing;
        // double enemyX = myX + distance * Math.sin(absBearing);
        // double enemyY = myY + distance * Math.cos(absBearing);
        double enemyHeading = e.getHeadingRadians();
        double enemyVelocity = e.getVelocity();
        double enemyAngle = Utils.normalRelativeAngle(absBearing - gunHeading);
        double firePower = 2.0;
        double bulletSpeed = 20 - 3 * firePower;

        String input = String.format(Locale.US, "%.2f,%.2f,%.4f,%.2f,%.4f,%.2f,%.4f,%.2f,%.2f",
                myX, myY, gunHeading, distance, enemyHeading,
                enemyVelocity, enemyAngle, firePower, bulletSpeed);

        try {
            out.println(input);
            String response = in.readLine();
            System.out.println("[JAVA] Model powiedział: " + response);

            // Celuj zawsze, nie tylko gdy strzelasz
            setTurnGunRightRadians(enemyAngle);

            if ("1".equals(response)) {
                if (getGunHeat() == 0) {
                    fire(firePower);
                    System.out.println("[JAVA] STRZAŁ 🔥");
                }
            }
        } catch (IOException ex) {
            System.out.println("[JAVA] Błąd komunikacji: " + ex.getMessage());
        }

        // Śledź przeciwnika radarem
        double radarTurn = Utils.normalRelativeAngle(absBearing - getRadarHeadingRadians());
        setTurnRadarRightRadians(radarTurn);

        // Ruch w stronę przeciwnika
        setTurnRightRadians(Utils.normalRelativeAngle(absBearing - getHeadingRadians()));
        setAhead(distance - 100);

        execute();
    }

    public void onDeath(DeathEvent e) {
        closeSocket();
    }

    public void onWin(WinEvent e) {
        closeSocket();
    }

    private void connectToPythonServer() {
        try {
            socket = new Socket("localhost", 5001);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            System.out.println("[JAVA] Połączono z serwerem Pythona");
        } catch (IOException e) {
            System.out.println("[JAVA] Błąd połączenia z serwerem: " + e.getMessage());
        }
    }

    private void closeSocket() {
        try {
            if (out != null) out.close();
            if (in != null) in.close();
            if (socket != null) socket.close();
        } catch (IOException ex) {
            System.out.println("[JAVA] Błąd zamykania socketu: " + ex.getMessage());
        }
    }
}
