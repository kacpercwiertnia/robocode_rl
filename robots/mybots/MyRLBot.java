// API 1.1
// Non-Sandboxed robot
package mybots;

import robocode.*;
import java.io.*;

public class MyRLBot extends AdvancedRobot {

    private double rewardValue = 0.0;

    public void run() {
        // Odseparowanie radaru i działa od obrotu robota
        setAdjustRadarForGunTurn(true);
        setAdjustGunForRobotTurn(true);

        // Obracaj radar nieskończenie, by wymuszać skanowanie
        while (true) {
            setTurnRadarRight(Double.POSITIVE_INFINITY);
            execute();
        }
    }

    public void onScannedRobot(ScannedRobotEvent e) {
        try {
            out.println("== onScannedRobot START ==");
            // === Dane wejściowe dla agenta RL ===
            double myX = getX();
            double myY = getY();
            double gunHeading = getGunHeadingRadians();
            double enemyDistance = e.getDistance();
            double enemyBearing = e.getBearingRadians();
            double enemyVelocity = e.getVelocity();
            double enemyHeading = e.getHeadingRadians();

            double absoluteBearing = getHeadingRadians() + enemyBearing;
            double relativeAngle = normalizeAngle(absoluteBearing - gunHeading);

            double dummyPower = 1.0;
            double bulletSpeed = 20 - 3 * dummyPower;

            File tempFile = new File("io/state.tmp");
            File realFile = new File("io/state.csv");

            try (PrintWriter writer = new PrintWriter(new FileWriter("io/state.csv"))) {
                writer.println(myX + "," + myY + "," + gunHeading + "," +
                            enemyDistance + "," + enemyHeading + "," + enemyVelocity + "," +
                            relativeAngle + "," + dummyPower + "," + bulletSpeed);
            } catch (IOException ex) {
                out.println("Błąd zapisu state.csv: " + ex.getMessage());
            }

            tempFile.renameTo(realFile);

            // === Odbierz decyzję (siłę strzału) ===
            double decisionPower = 1.0; // domyślna wartość
            try {
                File decisionFile = new File("io/decision.txt");
                while (!decisionFile.exists()) {
                    Thread.sleep(5);
                }

                try (BufferedReader reader = new BufferedReader(new FileReader(decisionFile))) {
                    String line = reader.readLine();
                    if (line != null && !line.trim().isEmpty()) {
                        decisionPower = Double.parseDouble(line.trim());
                    } else {
                        out.println("Plik decision.txt pusty, używam domyślnej siły 1.0");
                        decisionPower = 1.0;
                    }
                }

                decisionFile.delete();

            } catch (Exception ex) {
                out.println("Błąd odczytu decision.txt: " + ex.getMessage());
                decisionPower = 1.0;
            }

            // === Zabezpieczenie siły ognia ===
            if (decisionPower < 0.1 || decisionPower > 3.0 || Double.isNaN(decisionPower)) {
                out.println("Nieprawidłowa siła ognia: " + decisionPower + ", ustawiono domyślną 1.0");
                decisionPower = 1.0;
            }

            // === Ustawienie działa i strzał ===
            turnGunRightRadians(relativeAngle);
            Bullet bullet = fireBullet(decisionPower);
            if (bullet == null) {
                out.println("Nie udało się oddać strzału – prawdopodobnie za mało energii.");
            }
            out.println("== onScannedRobot END ==");
        } catch (Exception ex) {
            out.println("🔥 BŁĄD GŁÓWNY: " + ex.getClass().getName() + " – " + ex.getMessage());
            ex.printStackTrace(out);
        }
    }

    public void onBulletHit(BulletHitEvent e) {
        rewardValue = 1.0;
        saveReward();
    }

    public void onBulletMissed(BulletMissedEvent e) {
        rewardValue = -0.2;
        saveReward();
    }

    private void saveReward() {
        try (PrintWriter writer = new PrintWriter(new FileWriter("io/reward.txt"))) {
            writer.println(rewardValue);
        } catch (IOException ex) {
            out.println("Błąd zapisu reward.txt: " + ex.getMessage());
        }
    }

    private double normalizeAngle(double angle) {
        while (angle > Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }
}
