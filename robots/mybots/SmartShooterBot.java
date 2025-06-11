package mybots;

import robocode.*;
import robocode.util.Utils;

import java.io.*;
import java.net.Socket;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

import mybots.DataCollectingBot.BulletData;

public class SmartShooterBot extends AdvancedRobot {
    private Socket socket;
    private PrintWriter out;
    private PrintWriter writer;
    private BufferedReader in;
    private Map<Bullet, BulletData> bulletMap = new HashMap<>();

    public void run() {
        File dataFilePath = new File(getDataDirectory(), "smart_battle_data.csv");
        boolean fileExists = dataFilePath.exists();

        try {
            writer = new PrintWriter(new BufferedWriter(new FileWriter(getDataFile("smart_battle_data.csv"), true)));

            if (!fileExists){
                writer.println("bot_x,bot_y,gun_heading,distance,enemy_heading,enemy_velocity,enemy_angle,fire_power,bullet_speed,hit");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

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
                    Bullet bullet = fireBullet(firePower);
                    if (bullet != null) {
                        bulletMap.put(bullet, new BulletData(myX, myY, gunHeading, distance,
                                enemyHeading, enemyVelocity, enemyAngle, firePower, bulletSpeed));
                        System.out.println("Oddano strzał!");
                    } else {
                        System.out.println("Nie udało się oddać strzału (bullet == null)");
                    }
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

    public void onBulletHit(BulletHitEvent e) {
        Bullet b = e.getBullet();
        BulletData data = bulletMap.get(b);
        if (data != null) {
            data.hit = 1;
            writer.println(data.toCsv());
            writer.flush();
            bulletMap.remove(b);
            System.out.println("Trafiono przeciwnika!");
        }
    }

    public void onBulletMissed(BulletMissedEvent e) {
        Bullet b = e.getBullet();
        BulletData data = bulletMap.get(b);
        if (data != null) {
            data.hit = 0;
            writer.println(data.toCsv());
            writer.flush();
            bulletMap.remove(b);
            System.out.println("Pudło!");
        }
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

    static class BulletData {
        double botX, botY, gunHeading;
        double distance, enemyHeading, enemyVelocity, enemyAngle;
        double firePower, bulletSpeed;
        int hit;

        public BulletData(double botX, double botY, double gunHeading, double distance,
                          double enemyHeading, double enemyVelocity, double enemyAngle,
                          double firePower, double bulletSpeed) {
            this.botX = botX;
            this.botY = botY;
            this.gunHeading = gunHeading;
            this.distance = distance;
            this.enemyHeading = enemyHeading;
            this.enemyVelocity = enemyVelocity;
            this.enemyAngle = enemyAngle;
            this.firePower = firePower;
            this.bulletSpeed = bulletSpeed;
            this.hit = -1;
        }

        public String toCsv() {
            return String.format(Locale.US, "%.2f,%.2f,%.4f,%.2f,%.4f,%.2f,%.4f,%.2f,%.2f,%d",
                    botX, botY, gunHeading, distance, enemyHeading,
                    enemyVelocity, enemyAngle, firePower, bulletSpeed, hit);
        }
    }
}
